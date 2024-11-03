import numpy as np
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange

from downstream.trainer import *


class Classification(Trainer):
    """ 
    A helper class for trajectory classification. 
    Class label is typically user or driver ID.
    """

    def __init__(self, **kwargs):
        super().__init__(task_name='classification', metric_type='classification', **kwargs)
        self.loss_func = F.cross_entropy

    def cal_label(self, label_meta):
        return torch.tensor(label_meta).long().to(self.device)


class Destination(Trainer):
    """ 
    A helper class for destination prediction. 
    Feeds the encoders with truncated trajectories, 
    then regard the destinations of trajectories (last point) as prediction target.
    """

    def __init__(self, pre_length, **kwargs):
        super().__init__(task_name='destination', metric_type='classification', **kwargs)
        self.pre_length = pre_length
        self.loss_func = F.cross_entropy

    def forward_encoders(self, *x):
        if len(x) < 2:
            return super().forward_encoders(*x)

        trip, valid_len = x[:2]
        return super().forward_encoders(trip, valid_len-self.pre_length, *x[2:])

    def cal_label(self, label_meta):
        if label_meta[0].dim() == 2:
            return label_meta[0][:, -1].long().detach()
        return label_meta[0][:, -1, 1].long().detach()


class TTE(Trainer):
    """ 
    A helper class for travel time estimation evaluation. 
    The prediction targets is the time span (in minutes) of trajectories.
    """

    def __init__(self, pre_length, **kwargs):
        super().__init__(task_name=f'tte', metric_type='regression', **kwargs)
        self.pre_length = pre_length
        self.loss_func = F.mse_loss

    def forward_encoders(self, *x):
        if len(x) < 2:
            return super().forward_encoders(*x)

        trip, valid_len = x[:2]
        return super().forward_encoders(trip, valid_len-self.pre_length, *x[2:])

    def cal_label(self, label_meta):
        return torch.tensor(label_meta).float().to(self.device)


class Search(Trainer):
    """
    A helper class for similar trajectory evaluation.
    """
    def __init__(self, sim_indices=[], **kwargs):
        super(Search, self).__init__(task_name=f'search', metric_type='classification', **kwargs)

        self.sim_indices = sim_indices

    def train(self):
        print("Similar Trajectory Search do not require training.")
        return self.models, self.predictor

    def cal_label(self, label_meta):
        return label_meta

    def prepare_batch_meta(self, batch_meta):
        zipped = list(zip(*batch_meta))
        enc_meta = []
        for i in self.enc_meta_i:
            meta_prepare_func = BatchPreparer.fetch_prepare_func(self.meta_types[i])
            enc_meta += meta_prepare_func(zipped[self.meta_split_i[i]:self.meta_split_i[i+1]], self.device)
        return enc_meta

    def prepare_sim_indices(self, select_set):
        assert len(self.sim_indices) == 1, "Only support one similarity meta now."

        sim_meta_type = self.sim_indices[0]
        qry_idx, tgt_idx, neg_idx = self.data.load_meta(sim_meta_type, select_set)

        return qry_idx.astype(int), tgt_idx.astype(int), neg_idx.astype(int)

    def eval(self, set_index, full_metric=True):
        set_name = SET_NAMES[set_index][1]
        self.prepare_batch_iter(set_index)
        self.eval_state()

        ex_meta = self.prepare_ex_meta(set_index)

        embeds = []
        for batch_meta in tqdm(next_batch(self.batch_iter, self.batch_size),
                               desc=f"Calculating embeds on {set_name} set",
                               total=self.num_iter, leave=False):
            batch_meta = self.prepare_batch_meta(batch_meta)
            encodes = self.forward_encoders(*batch_meta, *ex_meta)
            embeds.append(encodes.detach().cpu().numpy())
        embeds = np.concatenate(embeds, 0)

        qry_idx, tgt_idx, neg_idx = self.prepare_sim_indices(set_index)
        pres, labels = self.cal_pres_and_labels(embeds[qry_idx], embeds[tgt_idx], embeds[neg_idx])

        if full_metric:
            self.metric_and_save(labels, pres, set_name)
        else:
            if self.metric_type == 'regression':
                mape = mean_absolute_percentage_error(labels, pres)
                return mape, 1 / (mape + 1e-6)
            elif self.metric_type == 'classification':
                acc = accuracy_score(labels, pres.argmax(-1))
                return acc, acc

    def cal_pres_and_labels(self, query, target, negs):
        num_queries = query.shape[0]
        num_targets = target.shape[0]
        num_negs = negs.shape[0]
        assert num_queries == num_targets, "Number of queries and targets should be the same."

        query_t = repeat(query, 'nq d -> nq nt d', nt=num_targets)
        query_n = repeat(query, 'nq d -> nq nn d', nn=num_negs)
        target = repeat(target, 'nt d -> nq nt d', nq=num_queries)
        negs = repeat(negs, 'nn d -> nq nn d', nq=num_queries)

        dist_mat_qt = np.linalg.norm(query_t - target, ord=2, axis=2)
        dist_mat_qn = np.linalg.norm(query_n - negs, ord=2, axis=2)
        dist_mat = np.concatenate([dist_mat_qt[np.eye(num_queries).astype(bool)][:, None], dist_mat_qn], axis=1)

        pres = -1 * dist_mat

        labels = np.zeros(num_queries)

        return pres, labels
