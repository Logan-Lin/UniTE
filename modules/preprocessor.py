"""
All samplers should be able to accept multiple arguments, each argument is an tensor.
They should also return multiple values, with the number of values the same as the number of input arguments.
"""
import math
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
import networkx as nx
from einops import repeat
from sklearn import preprocessing
from utils import geo_distance, torch_delete


class Sampler(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name


class KHopSampler(Sampler):
    def __init__(self, jump, select):
        super().__init__(f'khop{jump}-{select}')

        self.jump = jump
        self.select = select

    def forward(self, *tensors):
        tensors = (tensor[:, self.select-1::self.jump] for tensor in tensors)
        return tensors


class PassSampler(Sampler):
    """ 
    As the name suggest, this sampler won't change the tensor given to it, only to output as it is. 
    """
    def __init__(self):
        super().__init__('pass')

    def forward(self, *tensors):
        return tensors


class IndexSampler(Sampler):
    def __init__(self, index):
        super().__init__(f'index{index}')
        self.index = index

    def forward(self, *tensors):
        tensors = (tensor[:, self.index] for tensor in tensors)
        return tensors


class PoolSampler(Sampler):
    def __init__(self, pool_method='mean'):
        super().__init__(f'pool{pool_method}')

        if pool_method == 'mean':
            self.pool_func = torch.mean
        elif pool_method == 'max':
            self.pool_func = torch.max
        elif pool_method == 'min':
            self.pool_func = torch.min
        else:
            raise NotImplementedError(pool_method)
    
    def forward(self, *tensors):
        tensors = (self.pool_func(tensor, dim=1) for tensor in tensors)
        return tensors


class MissingSampler(Sampler):
    def __init__(self, missing_rate=0.3):
        super().__init__(f'missing{missing_rate}')
        self.missing_rate = missing_rate

    def forward(self, *tensors):
        L = tensors[0].shape[1]
        missing_idx = np.random.choice(L, math.ceil(L * (1 - self.missing_rate)), replace=False)
        missing_idx.sort()
        missing_idx[0] = 0
        tensors = (tensor[:, missing_idx].detach() for tensor in tensors)
        return tensors


class SubsetSampler(Sampler):
    def __init__(self, subset_rate=0.7):
        super().__init__(f'subset{subset_rate}')
        self.subset_rate = subset_rate

    def forward(self, *tensors):
        # TODO: It's too weak now.
        # Modify src mask matrix only.
        out = []
        for tensor in tensors:
            if not isinstance(torch.flatten(tensor)[0].item(), bool):
                out.append(tensor)

            else:
                augs, aug_valid_lens = [], []
                for record in tensor:
                    length = (~record).sum()
                    record[min(math.ceil(length * self.subset_rate), len(record) - 1):] = True
                    augs.append(record)
                augs = torch.stack(augs, dim=0)
                out.append(augs.detach())
        return out


class RandomViewSampler(Sampler):
    """
    Only for trip and its corresponding tensors with shape (B, L, E)
    """
    def __init__(self, missing_ratio=0.3, subset_ratio=0.7, hop=2):
        super(RandomViewSampler, self).__init__(f'random_hop{hop}-missing{missing_ratio}-subset{subset_ratio}')

        self.missing_sampler = MissingSampler(missing_ratio)
        self.khop_sampler = KHopSampler(2, 1)
        self.subset_sampler = SubsetSampler(subset_ratio)
        self.pass_sampler = PassSampler()

    def forward(self, *tensors):
        flag = torch.rand(1).item()
        if flag < 1 / 4:
            return self.missing_sampler.forward(*tensors)
        elif flag < 2 / 4:
            return self.khop_sampler.forward(*tensors)
        elif flag < 3 / 4:
            return self.pass_sampler.forward(*tensors)
        else:
            return self.subset_sampler.forward(*tensors)


class Trajectory2VecSampler(Sampler):
    """
    processing the trip meta data is applied for model trajectory2vec. 
    """
    def __init__(self, windowsize=2, offset=1, seq_len=46):
        super().__init__(f'Trajectory2VecSampler-{windowsize}-{offset}-{seq_len}')
        self.windowsize = windowsize
        self.offset = offset
        self.seq_len = seq_len
        
    def completeTrajectories(self, simTrjss):
        '''
        Complete trajectory attributes
        Calculate distance, speed and rotation angle between adjacent points in trajectory based on latitude, longitude and time
        '''
        eps = 2.220446049250313e-16
        simTrjComps = []
        for simTrjs in simTrjss:
            trjsCom = []
            for i in range(0,len(simTrjs)):
                rec = []
                if i==0:
                    rec = [0,0,0,0]  # time, locationC (distance), speedC, rotC (rotation angle)
                else:
                    locC = geo_distance(simTrjs[i][1], simTrjs[i][2], simTrjs[i-1][1], simTrjs[i-1][2])  # Calculate distance between two points using lat/lon
                    rec.append(simTrjs[i][0])
                    rec.append(locC)
                    rec.append(locC/max((simTrjs[i][0]-simTrjs[i-1][0]), eps))  # Calculate speed using distance and time difference
                    rec.append(math.atan((simTrjs[i][2]-simTrjs[i-1][2])/ max((simTrjs[i][1]-simTrjs[i-1][1]), eps)))  # Calculate rotation angle using arctan
                trjsCom.append(rec)
            simTrjComps.append(trjsCom)
        return simTrjComps

    def computeFeas(self, simTrjCompss):
        '''
        Calculate trajectory features
        Calculate rate of distance change (distance diff/time), speed difference, and rotation angle difference between adjacent points
        '''
        eps = 2.220446049250313e-16
        simTrjFeas = []
        for simTrjComps in simTrjCompss:
            trjsComfea = []
            for i in range(0,len(simTrjComps)):
                rec = []
                if i==0:
                    rec = [0,0,0,0]  # time, locationC (distance), speedC, rotC (rotation angle)
                else:
                    locC = simTrjComps[i][1]
                    locCrate = locC/max((simTrjComps[i][0]-simTrjComps[i-1][0]), eps)
                    rec.append(simTrjComps[i][0])  # time
                    rec.append(locCrate)
                    rec.append(simTrjComps[i][2]-simTrjComps[i-1][2])
                    rec.append(simTrjComps[i][3]-simTrjComps[i-1][3])
                trjsComfea.append(rec)
            simTrjFeas.append(trjsComfea)
        return simTrjFeas

    def rolling_window(self, sample, windowsize, offset):
        '''
        Sliding window - split data into time slices to select records from trajectory
        '''
        timeLength = sample[len(sample)-1][0]
        windowLength = int (timeLength/offset)+1  # windows and offset are both time-based
        windows = []
        for i in range(0,windowLength):
            windows.append([])

        for record in sample:
            time = record[0]
            for i in range(0,windowLength):
                if (time>(i*offset)) & (time<(i*offset+windowsize)):  # 50% time overlap
                    windows[i].append(record)
        return windows

    def behavior_ext(self, windows):
        '''
        Behavior extraction
        For selected windows, use pandas describe() to calculate [mean, max, 75%, 50%, 25%, min] of [distance diff, speed diff, rotation diff]
        '''
        behavior_sequence = []
        for window in windows:
            behaviorFeature = []
            records = np.array(window)
            if len(records) != 0:
                p = pd.DataFrame(records)
                pdd =  p.describe()
                behaviorFeature.append(pdd[1][1])  # mean
                behaviorFeature.append(pdd[2][1])
                behaviorFeature.append(pdd[3][1])
                
                behaviorFeature.append(pdd[1][3])  # min
                behaviorFeature.append(pdd[2][3])
                behaviorFeature.append(pdd[3][3])

                behaviorFeature.append(pdd[1][4])  # 25%
                behaviorFeature.append(pdd[2][4])
                behaviorFeature.append(pdd[3][4])

                behaviorFeature.append(pdd[1][5])  # 50%
                behaviorFeature.append(pdd[2][5])
                behaviorFeature.append(pdd[3][5])

                behaviorFeature.append(pdd[1][6])  # 75%
                behaviorFeature.append(pdd[2][6])
                behaviorFeature.append(pdd[3][6])

                behaviorFeature.append(pdd[1][7])  # max
                behaviorFeature.append(pdd[2][7])
                behaviorFeature.append(pdd[3][7])

                behavior_sequence.append(behaviorFeature)
        return behavior_sequence

    def generate_behavior_sequences(self, sim_data):
        behavior_sequences = []
        for sample in sim_data:
            windows = self.rolling_window(sample, windowsize=self.windowsize, offset=self.offset)
            behavior_sequence = self.behavior_ext(windows)
            behavior_sequences.append(behavior_sequence)
        return behavior_sequences

    def generate_normal_behavior_sequence(self, behavior_sequences):
        behavior_sequences_normal = []
        templist = []
        for item in behavior_sequences:
            for ii in item:
                templist.append(ii)
        min_max_scaler = preprocessing.MinMaxScaler()
        templist_normal = min_max_scaler.fit_transform(templist).tolist()
        index = 0
        for item in behavior_sequences:
            behavior_sequence_normal = []
            for ii in item:
                behavior_sequence_normal.append(templist_normal[index])
                index = index + 1
            behavior_sequences_normal.append(behavior_sequence_normal)
        return behavior_sequences_normal

    def traj2vec_process_data(self, simTrjss):
        simTrjCompss = self.completeTrajectories(simTrjss)
        simTrjFeas = self.computeFeas(simTrjCompss)
        behavior_sequences = self.generate_behavior_sequences(simTrjFeas)
        behavior_sequences_normal = self.generate_normal_behavior_sequence(behavior_sequences)
        return behavior_sequences_normal
    
    @torch.no_grad()
    def forward(self, trip, valid_len):
        newtrip = []
        for i, l in enumerate(valid_len):
            tripi = trip[i, :l-1, :]
            newtrip.append(tripi)
        metas = (newtrip, valid_len)
        
        trips, trips_len = metas
        processed_trips = []
        for i, trip in enumerate(trips):
            seq = []
            for j, record in enumerate(trip):
                if(j < trips_len[i]):
                    record_new = record[[7, 3, 4]]  # 'seconds','lng','lat'
                    record_new[0] = record_new[0] - trip[0][7]
                    seq.append(record_new)
            processed_trips.append(torch.stack(seq))
        behavior_sequences_normal = self.traj2vec_process_data(processed_trips)
        defalt = torch.tensor([0 for _ in range(len(behavior_sequences_normal[0][0]))])  # Pad with zeros until all trajectories have seq_len * input_size features
        for seq in behavior_sequences_normal:
            while len(seq) < self.seq_len:
                seq.append(defalt)
        behavior_sequences_normal = torch.tensor(behavior_sequences_normal)
        return behavior_sequences_normal