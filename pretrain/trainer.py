import math
from time import time
from abc import abstractmethod

import pandas as pd
import numpy as np
import torch
from sklearn.utils import shuffle
from tqdm import tqdm, trange

from utils import create_if_noexists, next_batch, cal_model_size, BatchPreparer, MinMaxScaler, StandardScaler
from data import SET_NAMES
from pretrain.generative_losses import TrajectorySim, t2vec

class Trainer:
    """
    Base class of the pre-training helper class.
    Implements most of the functions shared by all types of pre-trainers.
    """

    def __init__(self, data, models, trainer_name,
                 loss_func, batch_size, num_epoch, lr, device,
                 log_name_key, cache_epoches=False, prop=1.0,
                 meta_types=['trip'], ex_meta_types=[], suffix='', **kwargs):
        """
        :param data: dataset object.
        :param models: list of models. Depending on the type of pretext task, they can be encoders or decoders.
        :param trainer_name: name of the pre-trainer.
        :param loss_func: loss function module defined by specific pretext task.
        :param log_name_key: the name key for saving training logs. All saved log will use this key as their file name.
        :param cache_epoches: whether to save all models after every training epoch.
        :param meta_types: list of meta types used for pre-training, corresponds to the meta names in dataset object.
        """
        self.data = data
        self.prop = prop
        # The list of models may have different usage in different types of trainers.
        self.models = [model.to(device) for model in models]
        self.trainer_name = trainer_name

        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.lr = lr
        self.device = device
        self.cache_epoches = cache_epoches
        if not isinstance(loss_func, list):
            loss_func = [loss_func]
        self.loss_func = [func.to(self.device) for func in loss_func]
        loss_name = '_'.join([func.name for func in loss_func])

        self.meta_types = meta_types
        self.ex_meta_types = ex_meta_types
        model_name = '_'.join([model.name for model in models])
        meta_name = '_'.join(self.meta_types)

        BASE_KEY = f'{trainer_name}_b{batch_size}-lr{lr}'
        if self.prop < 1:
            BASE_KEY += f'-prop{self.prop}'
        self.BASE_KEY = f'{BASE_KEY}{suffix}/{loss_name}/{self.data.name}_{meta_name}/{model_name}'
        self.model_cache_dir = f'{data.base_path}/model_cache/{self.BASE_KEY}'
        self.model_save_dir = f'{data.base_path}/model_save/{self.BASE_KEY}'
        self.log_save_dir = f'{data.base_path}/log/{self.BASE_KEY}'

        # if 'Word2Vec' in loss_name:
        #     self.optimizer = torch.optim.SparseAdam(self.gather_all_param(*self.models, *self.loss_func), lr=lr)
        # else:
        #     self.optimizer = torch.optim.Adam(self.gather_all_param(*self.models, *self.loss_func), lr=lr)

        self.optimizer = torch.optim.Adam(self.gather_all_param(*self.models, *self.loss_func), lr=lr)
        self.log_name_key = log_name_key

        for model in models + loss_func:
            print(model.name, 'size', cal_model_size(model), 'MB')

    def prepare_batch_iter(self, select_set):
        """ Prepare iterations of batches for one training epoch. """
        metas = []
        meta_lengths = [0]
        for meta_type in self.meta_types:
            meta = self.data.load_meta(meta_type, select_set)
            metas += meta
            meta_lengths.append(len(meta))
        self.meta_split_i = np.cumsum(meta_lengths)
        self.batch_iter = shuffle(list(zip(*metas)))
        self.batch_iter = self.batch_iter[:int(len(self.batch_iter) * self.prop)]
        self.num_iter = math.ceil((len(self.batch_iter) - 1) / self.batch_size)

    def prepare_batch_meta(self, batch_meta):
        """ Prepare seperated meta arguments for one batch. """
        zipped = list(zip(*batch_meta))
        prepared_metas = []
        for i, meta_type in enumerate(self.meta_types):
            meta_prepare_func = BatchPreparer.fetch_prepare_func(meta_type)
            prepared_metas += meta_prepare_func(zipped[self.meta_split_i[i]:self.meta_split_i[i+1]], self.device)
        return prepared_metas

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

    def train(self, start=-1):
        """
        Finish the full training process.

        :param start: if given a value of 0 or higher, will try to load the trained model 
            cached after the start-th epoch training, and resume training from there.
        """
        self.prepare_batch_iter(0)
        train_logs = self.train_epoches(start)
        self.save_models()

        # Save training logs to a HDF5 file.
        create_if_noexists(self.log_save_dir)
        train_logs.to_hdf(f'{self.log_save_dir}/{self.log_name_key}.h5', key='pretrain_log')

    def train_epoches(self, start=-1, desc='Pre-training'):
        """ Train the models for multiple iterations (denoted by num_epoch). """
        self.train_state()

        if start > -1:
            self.load_models(start)
            print('Resumed training from epoch', start)

        train_logs = []
        desc_text = f'{desc}, avg loss %.4f'
        with trange(start+1, self.num_epoch, desc=desc_text % 0.0) as tbar:
            for epoch_i in tbar:
                s_time = time()
                epoch_avg_loss = self.train_epoch(epoch_i)
                e_time = time()
                tbar.set_description(desc_text % epoch_avg_loss)
                train_logs.append([epoch_i, e_time - s_time, epoch_avg_loss])

                if self.cache_epoches and epoch_i < self.num_epoch - 1:
                    self.save_models(epoch_i)

        train_logs = pd.DataFrame(train_logs, columns=['epoch', 'time', 'loss'])
        return train_logs

    def train_epoch(self, epoch_i=None):
        """ Train the models for one epoch. """
        loss_log = []
        for batch_meta in tqdm(next_batch(shuffle(self.batch_iter), self.batch_size),
                               desc=f'-->Traverse batches', total=self.num_iter, leave=False):
            self.optimizer.zero_grad()
            loss = self.forward_loss(batch_meta)
            with torch.autograd.detect_anomaly():
                loss.backward(retain_graph=True)
            self.optimizer.step()

            loss_log.append(loss.item())
        return float(np.mean(loss_log))

    def finetune(self, **ft_params):
        for key, value in ft_params.items():
            if key in self.__dict__:
                setattr(self, key, value)
        self.prepare_batch_iter(0)
        train_logs = self.train_epoches(desc='Fine-tuning')

        create_if_noexists(self.log_save_dir)
        train_logs.to_hdf(f'{self.log_save_dir}/{self.log_name_key}.h5', key='finetune_log')

    @abstractmethod
    def forward_loss(self, batch_meta):
        """
        Controls how the trainer forward models and meta datas to the loss function.
        Might be different depending on specific type of pretex task.
        """
        return self.loss_func[0](self.models, *self.prepare_batch_meta(batch_meta))

    @staticmethod
    def gather_all_param(*models):
        """ Gather all learnable parameters in the models as a list. """
        parameters = []
        for encoder in models:
            parameters += list(encoder.parameters())
        return parameters

    def save_models(self, epoch=None):
        """ Save learnable parameters in the models as pytorch binaries. """
        for model in (*self.models, *self.loss_func):
            if epoch is not None:
                create_if_noexists(self.model_cache_dir)
                save_path = f'{self.model_cache_dir}/{model.name}_epoch{epoch}.model'
            else:
                create_if_noexists(self.model_save_dir)
                save_path = f'{self.model_save_dir}/{model.name}.model'
                print('Saved model', model.name)
            torch.save(model.state_dict(), save_path)

    def load_model(self, model, epoch=None):
        """ Load one of the model. """
        if epoch is not None:
            save_path = f'{self.model_cache_dir}/{model.name}_epoch{epoch}.model'
        else:
            save_path = f'{self.model_save_dir}/{model.name}.model'
        model.load_state_dict(torch.load(save_path, map_location=self.device))
        print('Load model', model.name)
        return model

    def load_models(self, epoch=None):
        """ 
        Load all models from file. 
        """
        for i, model in enumerate(self.models):
            self.models[i] = self.load_model(model, epoch)
        for i, model in enumerate(self.loss_func):
            self.loss_func[i] = self.load_model(model, epoch)

    def get_models(self):
        """ Obtain all models in the trainer in evluation state. """
        self.eval_state()
        return self.models

    def train_state(self):
        for model in self.models:
            model.train()
        for model in self.loss_func:
            model.train()

    def eval_state(self):
        for model in self.models:
            model.eval()
        for model in self.loss_func:
            model.eval()


class ContrastiveTrainer(Trainer):
    """
    Trainer for contrastive pre-training.
    """

    def __init__(self, contra_meta_i=[0, 0], **kwargs):
        """
        :param contra_meta_i: list of meta indices indicating which meta data to use for constrative training.
            The indices corresponds to the meta_types list.
        """
        super().__init__(trainer_name='contrastive',
                         suffix='_contra-' + ','.join(list(map(str, contra_meta_i))),
                         ** kwargs)
        self.contra_meta_i = contra_meta_i

    def forward_loss(self, batch_meta):
        metas = self.prepare_batch_meta(batch_meta)
        contra_meta = [metas[self.meta_split_i[i]:self.meta_split_i[i+1]]
                       for i in self.contra_meta_i]
        return self.loss_func[0](self.models, *contra_meta)


class GenerativeTrainer(Trainer):
    """
    Trainer for generative pre-training.
    Contains a generate function for evaluating the recovered input.
    """

    def __init__(self, enc_meta_i, rec_meta_i, **kwargs):
        """
        :param enc_meta_i: list of meta indices indicating which meta data to fed into the encoders.
        :param rec_meta_i: list of meta indices indicating which meta data to use as recovery target.
        """
        meta_i_name = [','.join(map(str, meta_i)) for meta_i in [enc_meta_i, rec_meta_i]]
        super().__init__(trainer_name='generative',
                         suffix=f'_enc{meta_i_name[0]}-rec{meta_i_name[1]}',
                         **kwargs)
        self.generation_save_dir = f'{self.data.base_path}/generation/{self.BASE_KEY}'
        self.enc_meta_i = enc_meta_i
        self.rec_meta_i = rec_meta_i

    def _prepare_enc_rec(self, batch_meta):
        metas = self.prepare_batch_meta(batch_meta)
        enc_meta, rec_meta = [], []
        for i in self.enc_meta_i:
            enc_meta += metas[self.meta_split_i[i]:self.meta_split_i[i+1]]
        for i in self.rec_meta_i:
            rec_meta += metas[self.meta_split_i[i]:self.meta_split_i[i+1]]
        return enc_meta, rec_meta

    def forward_loss(self, batch_meta):
        """ For generative training, the batch is split into encode and recovery meta, then fed into the loss function. """
        enc_meta, rec_meta = self._prepare_enc_rec(batch_meta)
        return self.loss_func[0](self.models, enc_meta, rec_meta)

    def generate(self, set_index, **gen_params):
        """ Generate and save recovered meta data. """
        for key, value in gen_params.items():
            if key in self.__dict__:
                setattr(self, key, value)

        self.prepare_batch_iter(set_index)
        self.eval_state()

        gen_dicts = []
        for batch_meta in tqdm(next_batch(self.batch_iter, self.batch_size),
                               desc='Generating', total=self.num_iter):
            enc_meta, rec_meta = self._prepare_enc_rec(batch_meta)
            gen_dict, gen_save_name = self.loss_func[0].generate(self.models, enc_meta, rec_meta,
                                                              stat=self.data.stat, **gen_params)
            gen_dicts.append(gen_dict)
        numpy_dict = {key: np.concatenate([gen_dict[key] for gen_dict in gen_dicts], 0) for key in gen_dicts[0].keys()}

        gen_save_dir = f'{self.generation_save_dir}/{gen_save_name}'
        create_if_noexists(gen_save_dir)
        gen_save_path = f'{gen_save_dir}/{SET_NAMES[set_index][1]}_{self.log_name_key}.npz'
        np.savez(gen_save_path, **numpy_dict)


class GenerativeIterationTrainer(GenerativeTrainer):
    def __init__(self, enc_meta_i, rec_meta_i, **kwargs):
        super().__init__(enc_meta_i, rec_meta_i, **kwargs)
        self.max_grad_norm = kwargs.get("max_grad_norm", None)

    def forward_loss(self, batch_meta):
        """ For generative training, the batch is split into encode and recovery meta, then fed into the loss function. """
        enc_meta, rec_meta = self._prepare_enc_rec(batch_meta)
        return self.loss_func[0](self.models, enc_meta, rec_meta, self.current_iteration_id)

    def train(self, start=-1):
        """
        Finish the full training process.

        :param start: if given a value of 0 or higher, will try to load the trained model
            cached after the start-th epoch training, and resume training from there.
        """
        self.prepare_batch_iter(0)
        self.current_iteration_id = self.num_iter * (start + 1) + 1
        # 整个训练过程只为disLoss设置一次数据生成器
        if isinstance(self.loss_func[0], (TrajectorySim, t2vec)):
            self.loss_func[0].set_disLoss_dataGenerator(self.batch_iter, self.batch_size)
        train_logs = self.train_epoches(start)
        self.save_models()

        # Save training logs to a HDF5 file.
        create_if_noexists(self.log_save_dir)
        train_logs.to_hdf(f'{self.log_save_dir}/{self.log_name_key}.h5', key='pretrain_log')

    def train_epoch(self, epoch_i=None):
        """ Train the models for one epoch. """
        loss_log = []
        # if isinstance(self.loss_func, TrajectorySim):
        #     self.loss_func.set_disLoss_dataGenerator(self.batch_iter, self.batch_size)
        for batch_meta in tqdm(next_batch(shuffle(self.batch_iter), self.batch_size),
                               desc=f'-->Traverse batches', total=self.num_iter, leave=False):
            self.optimizer.zero_grad()
            loss = self.forward_loss(batch_meta)
            self.current_iteration_id += 1
            loss.backward()
            # clip the gradients
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.gather_all_param(*self.models, *self.loss_func),
                                               self.max_grad_norm)
            self.optimizer.step()

            loss_log.append(loss.item())
        return float(np.mean(loss_log))


class MomentumTrainer(Trainer):
    """
    Trainer for momentum-style parameter updating.
    Requires the loss function contains extra "teacher" models symmetric to the base models.
    The parameters of the teacher models will be updated in a momentum-style.
    """

    def __init__(self, momentum, teacher_momentum, weight_decay, eps, warmup_epoch=10, **kwargs):
        super().__init__(trainer_name='momentum',
                         suffix=f'_m{momentum}-tm{teacher_momentum}-wd{weight_decay}-eps{eps}-we{warmup_epoch}',
                         **kwargs)

        self.momentum = momentum
        self.teacher_momentum = teacher_momentum
        self.warmup_epoch = warmup_epoch
        self.lamda = 1 / (kwargs['batch_size'] * eps / self.models[0].output_size)

        self.optimizer = torch.optim.SGD(self.gather_all_param(*self.models, *self.loss_func), lr=self.lr,
                                         momentum=momentum, weight_decay=weight_decay)

    def train(self, start=-1):
        self.prepare_batch_iter(0)
        # The schedules are used for controlling the learning rate, momentum, and lamda.
        self.momentum_schedule = self.cosine_scheduler(self.teacher_momentum, 1,
                                                       self.num_epoch, self.num_iter)
        self.lr_schedule = self.cosine_scheduler(self.lr, 0, self.num_epoch,
                                                 self.num_iter, warmup_epochs=self.warmup_epoch)
        self.lamda_schedule = self.lamda_scheduler(8/self.lamda, 1/self.lamda, self.num_epoch, self.num_iter,
                                                   warmup_epochs=self.warmup_epoch)
        self.train_epoches(start)

    def train_epoch(self, epoch_i):
        loss_log = []
        for batch_i, batch_meta in tqdm(enumerate(next_batch(shuffle(self.batch_iter), self.batch_size)),
                                        desc=f'{self.trainer_name} training {epoch_i+1}-th epoch',
                                        total=self.num_iter, leave=False):
            it = self.num_iter * epoch_i + batch_i
            cur_lr = self.lr_schedule[it]
            lamda_inv = self.lamda_schedule[it]
            momentum = self.momentum_schedule[it]

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = cur_lr

            self.optimizer.zero_grad()
            loss = self.loss_func[0](self.models, *self.prepare_batch_meta(batch_meta),
                                  lamda_inv=lamda_inv)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                for encoder, teacher in zip(self.models, self.loss_func[0].teachers):
                    for param_q, param_k in zip(encoder.parameters(), teacher.parameters()):
                        param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)

            loss_log.append(loss.item())
        return float(np.mean(loss_log))

    @staticmethod
    def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * niter_per_ep
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

        schedule = np.concatenate((warmup_schedule, schedule))
        assert len(schedule) == epochs * niter_per_ep
        return schedule

    @staticmethod
    def lamda_scheduler(start_warmup_value, base_value, epochs, niter_per_ep, warmup_epochs=5):
        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * niter_per_ep
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        schedule = np.ones(epochs * niter_per_ep - warmup_iters) * base_value
        schedule = np.concatenate((warmup_schedule, schedule))
        assert len(schedule) == epochs * niter_per_ep
        return schedule


class MultiTrainer(Trainer):
    """
    Obtain both contrastive and generative
    """
    def __init__(self, loss_func, loss_coef, contra_meta_i, gen_enc_meta_i, gen_rec_meta_i, **kwargs):
        self.common_meta_i = kwargs.get('common_meta_i', [])
        kwargs.pop('common_meta_i')

        super(MultiTrainer, self).__init__(trainer_name='multiple',
                                           suffix='_multi-' + \
                                           '_'.join([loss.name for loss in loss_func]),
                                           loss_func=loss_func,
                                           **kwargs)
        assert len(loss_coef) == len(loss_func), "Num of loss funcs must equal that of loss coeffs"
        assert len(loss_func) == 2, "Only one contrastive loss and one generative loss are supported now"
        # self.loss_func = [loss.to(self.device) for loss in loss_func]
        self.contra_loss_func = self.loss_func[0].to(self.device)
        self.gen_loss_func = self.loss_func[1].to(self.device)
        self.loss_coef = loss_coef

        self.contra_meta_i = contra_meta_i
        self.gen_enc_meta_i, self.gen_rec_meta_i = gen_enc_meta_i, gen_rec_meta_i

    def forward_loss(self, batch_meta):
        """ Calculate MLM loss and SimCLR at the same time """
        metas = self.prepare_batch_meta(batch_meta)

        contra_meta = [metas[self.meta_split_i[i]:self.meta_split_i[i+1]]
                       for i in self.contra_meta_i]
        enc_meta, rec_meta = [], []
        for i in self.gen_enc_meta_i:
            enc_meta += metas[self.meta_split_i[i]:self.meta_split_i[i + 1]]
        for i in self.gen_rec_meta_i:
            rec_meta += metas[self.meta_split_i[i]:self.meta_split_i[i + 1]]

        ex_metas = self.prepare_ex_meta(0)
        common_meta = []
        if self.common_meta_i:
            for i in self.common_meta_i:
                common_meta += ex_metas[self.ex_meta_split_i[i]:self.ex_meta_split_i[i + 1]]

        loss = 0
        loss += self.contra_loss_func(self.models, *contra_meta, common_meta=common_meta)
        loss += self.gen_loss_func(self.models, common_meta, enc_meta, rec_meta)

        return loss


class NoneTrainer():
    def __init__(self, models, data, device):
        self.models = [model.to(device) for model in models]
        self.BASE_KEY = f'end2end/none/{data.name}'
        self.device = device
        self.model_save_dir = f'{data.base_path}/model_save/{self.BASE_KEY}'

    def save_models(self):
        """ Save all models. """
        create_if_noexists(self.model_save_dir)
        for model in self.models:
            save_path = f'{self.model_save_dir}/{model.name}.model'
            torch.save(model.state_dict(), save_path)

    def load_model(self, model):
        """ Load one of the encoder. """
        save_path = f'{self.model_save_dir}/{model.name}.model'
        model.load_state_dict(torch.load(save_path, map_location=self.device))
        print('Load model from', save_path)
        return model

    def load_models(self):
        """ Load all encoders. """
        for i, model in enumerate(self.models):
            self.models[i] = self.load_model(model)

    def get_models(self):
        for model in self.models:
            model.eval()
        return self.models
    

class ADMMTrainer(GenerativeTrainer):
    '''
    Trainer applying the Alternating Direction of Method of Multipliers(ADMM).
    '''
    def __init__(self, enc_meta_i, rec_meta_i, lambda_=1.0, error=1.0e-7, epoch_stop_criteria=1.0e-2, **kwargs):
        super().__init__(enc_meta_i, rec_meta_i, **kwargs)
        self.optimizer = torch.optim.RMSprop(self.gather_all_param(*self.models, *self.loss_func), lr=self.lr)
        self.epoch_stop_criteria = epoch_stop_criteria
        self.lambda_ = lambda_
        self.error = error

    @staticmethod
    def ProximalMethod_l1(epsilon, x):
        output = np.zeros_like(x)
        output = (x - epsilon) * (x > epsilon) + \
                    (x + epsilon) * (x < -epsilon)

        return output
    
    def prepare_batch_iter(self, select_set):
        """
        Prepare iterations of batches for one training epoch.

        Attention: self.batch_iter is ndarray
        """
        metas = []
        meta_lengths = [0]
        LS = []
        for meta_type in self.meta_types:
            meta = self.data.load_meta(meta_type, select_set)
            # Initialize L_D, S to be zero matrices
            L_D = np.zeros_like(meta[0])
            S = np.zeros_like(meta[0])
            meta.append(L_D)
            meta.append(S)
            metas += meta
            meta_lengths.append(len(meta))
            # Initialize LS = XR
            LS += [meta[0]]
        self.meta_split_i = np.cumsum(meta_lengths)
        # self.meta_types中只有"robustDAA"的数据，可直接令self.batch_iter为ndarray类型
        # 包含num_samples个元素,每个元素形式为[src,tgt,L_D,S] (L_D和S会动态变化，因此不存为元组形式)
        self.batch_iter = np.array([list(item) for item in list(zip(*metas))])
        self.num_iter = math.ceil((len(self.batch_iter) - 1) / self.batch_size)
        self.LS = np.array([list(item) for item in list(zip(*LS))])
        # self.scaler = MinMaxScaler(self.batch_iter[:,0].max(),self.batch_iter[:,0].min())
        self.scaler = StandardScaler(self.batch_iter[:,0].mean(),self.batch_iter[:,0].std())
        
    def update_LD_S(self):
        '''Attention: self.batch_iter is ndarray'''
        print("Update L_D and S using model output.")
        new_LD = []
        for batch_meta in tqdm(next_batch(self.batch_iter, self.batch_size),
                               desc=f'-->Traverse batches', total=self.num_iter, leave=False):
            O_f = self.get_models_output(batch_meta).detach().cpu().numpy()
            new_LD.append(O_f)
        new_LD = np.concatenate(new_LD)

        self.batch_iter[:,-2] = self.scaler.inverse_transform(new_LD)
        S = self.batch_iter[:,0] - self.batch_iter[:,-2]
        self.batch_iter[:,-1] = self.ProximalMethod_l1(self.shrink, S)
        '''
        # self.batch_iter为list
        for i in trange(len(self.batch_iter), desc="Update L_D and S"):
            self.batch_iter[i][-2] = new_LD[i]
            S = self.batch_iter[i][0] - new_LD[i]
            self.batch_iter[i][-1] = self.ProximalMethod_l1(self.shrink, S)
        '''

    def train(self, start=-1):
        """
        Finish the full training process.

        :param start: if given a value of 0 or higher, will try to load the trained model 
            cached after the start-th epoch training, and resume training from there.
        
        Attention: self.batch_iter is ndarray
        """
        self.prepare_batch_iter(0)
        train_logs = []
        training_round = 0
        # mu = (X.size) / (4.0 * nplin.norm(self.X,1)) # X.size元素个数
        mu = (self.batch_iter[:,0].size) / (4.0 * np.linalg.norm(self.batch_iter[:,0],1))
        self.shrink = self.lambda_ / mu
        XFnorm = np.linalg.norm(self.batch_iter[:,0])

        while(True):
            print(f"\nTraining round {training_round+1}:")
            # 为L_D赋值
            self.batch_iter[:,-2] = self.batch_iter[:,0] - self.batch_iter[:,-1]
            
            # 训练模型
            train_log = self.train_epoches(start)
            train_logs.append(train_log)
            self.save_models()
            
            # 用模型输出更新L_D和S
            self.update_LD_S()

            # break criterion
            XR = self.batch_iter[:,0]
            L_D = self.batch_iter[:,-2]
            S = self.batch_iter[:,-1]
            ## break criterion 1: the L and S are close enough to X
            c1 = np.linalg.norm(XR - L_D - S) / XFnorm
            ## break criterion 2: there is no changes for L and S 
            c2 = np.linalg.norm(self.LS - L_D - S) / XFnorm
            if c1 < self.error or c2 < self.error :
                print("Training End")
                break

            ## save L + S for c2 check in the next iteration
            self.LS = L_D + S
            training_round += 1
    

        # Save training logs to a HDF5 file.
        create_if_noexists(self.log_save_dir)
        train_logs = pd.concat(train_logs)
        train_logs.to_hdf(f'{self.log_save_dir}/{self.log_name_key}.h5', key='pretrain_log')
    
    def train_epoches(self, start=-1, desc='Pre-training'):
        """ Train the models for multiple iterations (denoted by num_epoch). """
        self.train_state()

        if start > -1:
            self.load_models(start)
            print('Resumed training from epoch', start)

        train_logs = []
        desc_text = f'{desc}, avg loss %.4f'
        with trange(start+1, self.num_epoch, desc=desc_text % 0.0) as tbar:
            for epoch_i in tbar:
                s_time = time()
                epoch_avg_loss = self.train_epoch(epoch_i)
                e_time = time()
                tbar.set_description(desc_text % epoch_avg_loss)
                train_logs.append([epoch_i, e_time - s_time, epoch_avg_loss])

                if self.cache_epoches and epoch_i < self.num_epoch - 1:
                    self.save_models(epoch_i)
                
                # 满足停止条件，直接返回
                if epoch_avg_loss <= self.epoch_stop_criteria:
                    train_logs = pd.DataFrame(train_logs, columns=['epoch', 'time', 'loss'])
                    return train_logs
                
        train_logs = pd.DataFrame(train_logs, columns=['epoch', 'time', 'loss'])
        return train_logs
    
    # def forward_loss(self, batch_meta):
    #     """ For generative training, the batch is split into encode and recovery meta, then fed into the loss function. """
    #     enc_meta, rec_meta = self._prepare_enc_rec(batch_meta)
    #     return self.loss_func[0](self.models, enc_meta, rec_meta, self.scaler)
    
    def get_models_output(self, batch_meta):
        enc_meta, _ = self._prepare_enc_rec(batch_meta)
        return self.loss_func[0].EncoderDecoder(self.models, enc_meta)

