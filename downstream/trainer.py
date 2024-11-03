import math
from abc import abstractmethod

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

from utils import create_if_noexists, cal_classification_metric, cal_regression_metric, next_batch, mean_absolute_percentage_error, BatchPreparer
from data import SET_NAMES
from pretrain.trainer import Trainer as PreTrainer


class Trainer:
    """
    Base class of the downstream trainer.
    Implements most of the functions shared by various downstream tasks.
    """
    def __init__(self, task_name, base_name, metric_type, data, models,
                 predictor, batch_size, num_epoch, lr, device, log_name_key,
                 meta_types=['trip'], ex_meta_types=[], enc_meta_i=[0], label_meta_i=[0],
                 es_epoch=-1, finetune=False, save_prediction=False):
        self.task_name = task_name
        self.metric_type = metric_type

        self.data = data
        # All models feed into the downstream trainer will be used for calculating the embedding vectors.
        # The embedding vectors will be concatenated along the feature dimension.
        self.models = [model.to(device) for model in models]
        # The predictor is fed with the embedding vectors, and output the prediction required by the downstream task.
        self.predictor = predictor.to(device)

        self.batch_size = batch_size
        self.es_epoch = es_epoch
        self.num_epoch = num_epoch
        self.lr = lr
        self.device = device
        self.log_name_key = log_name_key

        self.finetune = finetune
        self.save_prediction = save_prediction
        self.meta_types = meta_types  # The first type of meta must be trip.
        self.ex_meta_types = ex_meta_types  # Extra metas, currently for meta data that do not vary across batches.
        self.enc_meta_i = enc_meta_i
        self.label_meta_i = label_meta_i

        model_name = '_'.join([f'{model.name}-ds-{model.sampler.name}' for model in models])
        self.base_key = f'{task_name}/{base_name}/{model_name}_ft{int(finetune)}'
        self.model_cache_dir = f'{data.base_path}/model_cache/{self.base_key}'
        self.model_save_dir = f'{data.base_path}/model_save/{self.base_key}'
        self.log_save_dir = f'{data.base_path}/log/{self.base_key}'
        self.pred_save_dir = f'{data.base_path}/pred/{self.base_key}'

        self.optimizer = torch.optim.Adam(PreTrainer.gather_all_param(*self.models, self.predictor), lr=lr)

    def prepare_batch_iter(self, select_set):
        metas = []
        meta_lengths = [0]
        for meta_type in self.meta_types:
            meta = self.data.load_meta(meta_type, select_set)
            metas += meta
            meta_lengths.append(len(meta))
        self.meta_split_i = np.cumsum(meta_lengths)
        self.batch_iter = list(zip(*metas))
        self.num_iter = math.ceil((len(self.batch_iter) - 1) / self.batch_size)

    def prepare_batch_meta(self, batch_meta):
        zipped = list(zip(*batch_meta))
        enc_meta, label_meta = [], []
        for i in self.enc_meta_i:
            meta_prepare_func = BatchPreparer.fetch_prepare_func(self.meta_types[i])
            enc_meta += meta_prepare_func(zipped[self.meta_split_i[i]:self.meta_split_i[i+1]], self.device)
        for i in self.label_meta_i:
            meta_prepare_func = BatchPreparer.fetch_prepare_func(self.meta_types[i])
            label_meta += meta_prepare_func(zipped[self.meta_split_i[i]:self.meta_split_i[i+1]], self.device)
        label = self.cal_label(label_meta)
        return enc_meta, label

    def prepare_ex_meta(self, select_set):
        ex_metas = []
        ex_meta_lengths = [0]
        for ex_meta_type in self.ex_meta_types:
            if ex_meta_type == 'transprob':
                select_set = 0
            ex_meta = self.data.load_meta(ex_meta_type, select_set)
            meta_prepare_func = BatchPreparer.fetch_prepare_func(ex_meta_type)
            ex_metas += meta_prepare_func(ex_meta, self.device)
            ex_meta_lengths.append(len(ex_meta))
        self.ex_meta_split_i = np.cumsum(ex_meta_lengths)
        return ex_metas

    def train(self):
        num_noimprove_epoches = 0
        best_metric = 0.0
        train_logs = []
        desc_text = 'Downstream training, val metric %.4f'
        with trange(self.num_epoch, desc=desc_text % 0.0) as tbar:
            for epoch_i in tbar:
                train_loss = self.train_epoch()
                val_metric, es_metric = self.eval(1, full_metric=False)
                train_logs.append([epoch_i, val_metric, train_loss])
                tbar.set_description(desc_text % val_metric)

                if epoch_i == 0:
                    self.save_models('best')
                if self.es_epoch > -1:
                    if es_metric > best_metric:
                        best_metric = es_metric
                        num_noimprove_epoches = 0
                        self.save_models('best')
                    else:
                        num_noimprove_epoches += 1
                if self.es_epoch > -1 and num_noimprove_epoches >= self.es_epoch:
                    self.load_models('best')
                    tbar.set_description('Early stopped')
                    break

        self.save_models()
        # Save training logs to a HDF5 file.
        create_if_noexists(self.log_save_dir)
        train_logs = pd.DataFrame(train_logs, columns=['epoch', 'val_metric', 'loss'])
        train_logs.to_hdf(f'{self.log_save_dir}/{self.log_name_key}.h5', key='downstream_train_log')

        return self.models, self.predictor

    def train_epoch(self):
        self.prepare_batch_iter(0)
        self.train_state()

        loss_log = []
        for batch_meta in tqdm(next_batch(shuffle(self.batch_iter), self.batch_size),
                               desc=f'-->Traverse batches', total=self.num_iter, leave=False):
            enc_meta, label = self.prepare_batch_meta(batch_meta)
            ex_meta = self.prepare_ex_meta(0)
            encodes = self.forward_encoders(*enc_meta, *ex_meta)
            pre = self.predictor(encodes).squeeze(-1)

            loss = self.loss_func(pre, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_log.append(loss.item())
        return float(np.mean(loss_log))

    def eval(self, set_index, full_metric=True):
        set_name = SET_NAMES[set_index][1]
        self.prepare_batch_iter(set_index)
        self.eval_state()

        pres, labels = [], []
        for batch_meta in tqdm(next_batch(self.batch_iter, self.batch_size),
                               desc=f'Evaluating on {set_name} set',
                               total=self.num_iter, leave=False):
            enc_meta, label = self.prepare_batch_meta(batch_meta)
            ex_meta = self.prepare_ex_meta(0)
            encodes = self.forward_encoders(*enc_meta, *ex_meta)
            pre = self.predictor(encodes).squeeze(-1)

            pres.append(pre.detach().cpu().numpy())
            labels.append(label.cpu().numpy())
        pres, labels = np.concatenate(pres, 0), np.concatenate(labels, 0)

        if full_metric:
            self.metric_and_save(labels, pres, set_name)
        else:
            if self.metric_type == 'regression':
                mape = mean_absolute_percentage_error(labels, pres)
                return mape, 1 / (mape + 1e-6)
            elif self.metric_type == 'classification':
                acc = accuracy_score(labels, pres.argmax(-1))
                return acc, acc

    @abstractmethod
    def cal_label(self, label_meta):
        pass

    def loss_func(self, pre, label):
        pass

    def forward_encoders(self, *x):
        """ Feed the input to all encoders and concatenate the embedding vectors.  """
        encodes = [encoder(*x) for encoder in self.models]
        if not self.finetune:
            encodes = [encode.detach() for encode in encodes]
        encodes = torch.cat(encodes, -1)
        return encodes  # (B, num_encoders * E)

    def train_state(self):
        """ Turn all models and the predictor into training mode.  """
        for encoder in self.models:
            encoder.train()
        self.predictor.train()

    def eval_state(self):
        """ Turn all models and the predictor into evaluation mode.  """
        for encoder in self.models:
            encoder.eval()
        self.predictor.eval()

    def save_models(self, epoch=None):
        """ Save the encoder model and the predictor model. """
        for model in (*self.models, self.predictor):
            if epoch is not None:
                create_if_noexists(self.model_cache_dir)
                save_path = f'{self.model_cache_dir}/{model.name}_epoch{epoch}.model'
            else:
                create_if_noexists(self.model_save_dir)
                save_path = f'{self.model_save_dir}/{model.name}.model'
                print('Saved model', model.name)
            torch.save(model.state_dict(), save_path)

    def load_model(self, model, epoch=None):
        """ Load one of the encoder. """
        if epoch is not None:
            save_path = f'{self.model_cache_dir}/{model.name}_epoch{epoch}.model'
        else:
            save_path = f'{self.model_save_dir}/{model.name}.model'
            print('Load model', model.name)
        model.load_state_dict(torch.load(save_path, map_location=self.device))
        return model

    def load_models(self, epoch=None):
        """ Load all encoders. """
        for i, encoder in enumerate(self.models):
            self.models[i] = self.load_model(encoder, epoch)
        self.predictor = self.load_model(self.predictor, epoch)

    def metric_and_save(self, labels, pres, save_name):
        """ Calculate the evaluation metric, then save the metric and the prediction result. """
        if self.metric_type == 'classification':
            metric = cal_classification_metric(labels, pres)
        elif self.metric_type == 'regression':
            metric = cal_regression_metric(labels, pres)
        else:
            raise NotImplementedError(f'No type "{type}".')
        print(metric)

        create_if_noexists(self.log_save_dir)
        metric.to_hdf(f'{self.log_save_dir}/{save_name}_{self.log_name_key}.h5',
                      key='metric', format='table')

        if self.save_prediction:
            create_if_noexists(self.pred_save_dir)
            np.savez(f'{self.pred_save_dir}/{save_name}_{self.log_name_key}.npz',
                     labels=labels, pres=pres)
