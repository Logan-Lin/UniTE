import math
import os
import shutil
from heapq import heappush, heappop
from itertools import count, islice
from os.path import exists
from datetime import datetime
from collections import namedtuple

import pandas as pd
import numpy as np
import torch
from torchvision.transforms import transforms
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, recall_score, accuracy_score, roc_auc_score
from tqdm import trange
from einops import rearrange
import networkx as nx


image_preprocessing = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((64, 64)),
                        transforms.ToTensor()
                    ])


def idDocker():
    return os.path.exists('/.dockerenv')


def create_if_noexists(*paths):
    """ Create directories if they don't exist already. """
    for path in paths:
        if not exists(path):
            os.makedirs(path)


def remove_if_exists(path):
    """ Remove a file if it exists. """
    if exists(path):
        os.remove(path)


def next_batch(data, batch_size):
    """ Yield the next batch of given data. """
    data_length = len(data)
    num_batches = math.ceil(data_length / batch_size)
    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_length)
        if end_index - start_index > 1:
            yield data[start_index:end_index]


def next_batch_recycle(data, batch_size):
    """ Yield the next batch of given data, and it can recycle"""
    assert batch_size > 0
    data_length = len(data)
    assert batch_size <= data_length, 'Batch size is large than total data length. ' \
                                      'Please check your data or change batch size.'
    batch_index = 0
    while True:
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_length)
        # if end_index - start_index > 1:
        if end_index - start_index == batch_size:
            yield data[start_index:end_index]
            batch_index += 1
        else:  # recycle(no use the last incomplete batch)
            batch_index = 0


def clean_dirs(*dirs):
    """ Remove the given directories, including all contained files and sub-directories. """
    for dir in dirs:
        if os.path.exists(dir):
            shutil.rmtree(dir)


def mean_absolute_percentage_error(y_true, y_pred):
    """ Calculcates the MAPE metric. """
    mape = np.mean(np.abs((y_true - y_pred) / y_true))
    return mape


def cal_regression_metric(label, pres):
    """ Calculcate all common regression metrics. """
    rmse = math.sqrt(mean_squared_error(label, pres))
    mae = mean_absolute_error(label, pres)
    mape = mean_absolute_percentage_error(label, pres)

    s = pd.Series([rmse, mae, mape], index=['rmse', 'mae', 'mape'])
    return s


def top_n_accuracy(truths, preds, n):
    """ Calculcate Acc@N metric. """
    best_n = np.argsort(preds, axis=1)[:, -n:]
    successes = 0
    for i, truth in enumerate(truths):
        if truth in best_n[i, :]:
            successes += 1
    return float(successes) / truths.shape[0]


def cal_classification_metric(labels, pres):
    """
    Calculates all common classification metrics.

    :param labels: classification label, with shape (N).
    :param pres: predicted classification distribution, with shape (N, num_class).
    """
    pres_index = pres.argmax(-1)  # (N)
    macro_f1 = f1_score(labels, pres_index, average='macro', zero_division=0)
    macro_recall = recall_score(labels, pres_index, average='macro', zero_division=0)
    acc = accuracy_score(labels, pres_index)
    n_list = [5, 10]
    top_n_acc = [top_n_accuracy(labels, pres, n) for n in n_list]

    s = pd.Series([macro_f1, macro_recall, acc] + top_n_acc,
                  index=['macro_f1', 'macro_rec'] +
                  [f'acc@{n}' for n in [1] + n_list])
    return s


def intersection(lst1, lst2):
    """ Calculates the intersection of two sets, or lists. """
    lst3 = list(set(lst1) & set(lst2))
    return lst3


def get_datetime_key():
    """ Get a string key based on current datetime. """
    return 't' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


class BatchPreparer:
    """
    Prepare functions for different types of batch data.
    """

    @staticmethod
    def fetch_prepare_func(meta_type):
        """ Fetch a specific prepare function based on the meta type. """
        if 'timemat' in meta_type:
            return BatchPreparer.prepare_timemat_batch
        elif 'traj2vectime' in meta_type:
            return BatchPreparer.prepare_traj2vec_batch
        elif meta_type == 'trip' or 'resample' in meta_type or 'mlm' in meta_type or meta_type == 'fromto' or \
                'trim' in meta_type or 'shift' in meta_type or 'quadkey' in meta_type or 'speedacc' in meta_type:
            return BatchPreparer.prepare_trip_batch
        elif 'cand' in meta_type:
            return BatchPreparer.prepare_cand_batch
        elif 'glm' in meta_type:
            return BatchPreparer.prepare_glm_batch
        elif 'trajsim' in meta_type or 't2vec' in meta_type:
            return BatchPreparer.prepare_trajsim_batch
        elif 'transprob' in meta_type:
            return BatchPreparer.prepare_transprob
        elif 'trajimage' in meta_type:
            return BatchPreparer.prepare_trajimage_batch
        elif 'detourtgt' in meta_type or 'detourqry' in meta_type or \
            'hoptgt' in meta_type or 'hopqry' in meta_type or 'hopneg' in meta_type:
            return BatchPreparer.prepare_detourtgt
        elif 'detourneg' in meta_type:
            return BatchPreparer.prepare_detourneg
        elif meta_type == 'class':
            return BatchPreparer.prepare_class_batch
        elif meta_type == 'tte':
            return BatchPreparer.prepare_tte_batch
        elif 'slice' in meta_type:
            return BatchPreparer.prepare_slice_batch
        elif 'timefea' in meta_type or 'classbatch' in meta_type:
            return BatchPreparer.prepare_timefea_batch
        elif 'robustDAA' in meta_type:
            return BatchPreparer.prepare_robustDAA_batch
        else:
            raise NotImplementedError('No prepare function for meta type "' + meta_type + '".')

    @staticmethod
    def prepare_timemat_batch(batch_meta, device):
        """
        Prepare a batch for trips that obtain time mat
        """
        time_mat = np.stack(batch_meta, 0)
        time_mat = torch.from_numpy(time_mat).float().to(device)

        return time_mat

    @staticmethod
    def prepare_trip_batch(batch_meta, device):
        """ 
        Prepare a batch for "trip" type of meta data.
        This type of meta data all contains a trajectory and a valid length arguments.
        """
        batch_trip, valid_len = batch_meta
        batch_trip = np.stack(batch_trip, 0)
        batch_trip = torch.from_numpy(batch_trip).float().to(device)
        valid_len = torch.tensor(valid_len).long().to(device)

        return batch_trip, valid_len

    @staticmethod
    def prepare_traj2vec_batch(batch_meta, device):
        """
        Prepare a batch for "traj2vec" type of meta data.
        """
        batch_trip, valid_len, batch_time = batch_meta
        batch_trip = np.stack(batch_trip, 0)
        batch_time = np.stack(batch_time, 0)

        batch_trip = torch.from_numpy(batch_trip).float().to(device)
        valid_len = torch.tensor(valid_len).long().to(device)
        batch_time = torch.from_numpy(batch_time).float().to(device)

        return batch_trip, valid_len, batch_time

    @staticmethod
    def prepare_trajimage_batch(batch_meta, device):
        batch_meta = batch_meta[0]
        batch_meta = [image_preprocessing(rearrange(ima, 'c h w -> h w c').astype(np.float32))
                      for ima in batch_meta]
        batch_meta = torch.stack(batch_meta, 0).float().to(device)

        return batch_meta,

    @staticmethod
    def prepare_detourtgt(batch_meta, device):
        tgt_detoured_trips, tgt_detour_lens = batch_meta[:2]

        tgt_detoured_trips = torch.from_numpy(np.stack(tgt_detoured_trips, 0)).float().to(device)
        tgt_detour_lens = torch.tensor(tgt_detour_lens).long().to(device)
        tgt_detoured_trips[:, :, 7] = torch.round(tgt_detoured_trips[:, :, 7] / 60) % 1440

        return tgt_detoured_trips, tgt_detour_lens

    @staticmethod
    def prepare_detourneg(ex_meta, device):
        neg_detoured_trips, neg_detour_lens = ex_meta

        neg_detoured_trips = torch.from_numpy(np.stack(neg_detoured_trips, 0)).float().to(device)
        neg_detour_lens = torch.tensor(neg_detour_lens).long().to(device)

        return neg_detoured_trips, neg_detour_lens

    @staticmethod
    def prepare_cand_batch(batch_meta, device):
        """
        Prepare a batch for "candidate" type of meta data.
        This type of meta data is formed by boolean array.
        """
        batch_cand = (torch.from_numpy(np.stack(cand, 0)).long().to(device) for cand in batch_meta)
        return batch_cand

    @staticmethod
    def prepare_class_batch(batch_meta, device):
        """
        Prepare a batch for "class" type of meta data.
        """
        batch_class = torch.tensor(batch_meta[0]).reshape(-1, 1).long().to(device)
        return batch_class

    @staticmethod
    def prepare_timefea_batch(batch_meta, device):
        """
        Prepare a batch for "timefea" type of meta data.
        """
        batch_class = torch.tensor(batch_meta[0]).long().to(device)
        return batch_class,

    @staticmethod
    def prepare_glm_batch(batch_meta, device):
        trip, trip_pos, src, tgt, pos, trip_len, src_len = batch_meta
        trip, trip_pos, src, tgt, pos = [torch.from_numpy(np.stack(a, 0)).float().to(device)
                                         for a in [trip, trip_pos, src, tgt, pos]]
        trip_len, src_len = [torch.tensor(a).long().to(device) for a in [trip_len, src_len]]

        return trip, trip_pos, src, tgt, pos, trip_len, src_len

    @staticmethod
    def prepare_trajsim_batch(batch_meta, device):
        """
        Prepare a batch for "trajsim" or 't2vec' type of meta data.
        """
        src_arr, tgt_arr, mta_arr = batch_meta
        # 获取一批补位后的数据对象 TF=['src', 'lengths', 'trg', 'invp']
        # src (batch, seq_len1), lengths (batch,), trg (batch, seq_len2)
        gen_data = pad_arrays_pair(src_arr, tgt_arr, keep_invp=False)
        src, lengths, trg = gen_data.src.to(device), gen_data.lengths.to(device), gen_data.trg.to(device)

        return src, lengths, trg

    @staticmethod
    def prepare_transprob(ex_meta, device):
        edge_index, trans_prob = ex_meta
        edge_index = torch.from_numpy(edge_index).long().to(device)
        trans_prob = torch.from_numpy(trans_prob).float().to(device)
        return edge_index, trans_prob

    @staticmethod
    def prepare_tte_batch(batch_meta, device):
        """
        Prepare a batch for "class" type of meta data.
        """
        batch_class = torch.tensor(batch_meta[0]).reshape(-1, 1).float().to(device)
        return batch_class

    @staticmethod
    def prepare_slice_batch(batch_meta, device):
        """
        Prepare positive, negative samples for geospatial constrains
        """
        sources, destinations, midways, negs = batch_meta
        sources = torch.from_numpy(np.stack(sources, 0)).long().to(device)
        destinations = torch.from_numpy(np.stack(destinations, 0)).long().to(device)
        midways = torch.from_numpy(np.stack(midways, 0)).long().to(device)
        negs = torch.from_numpy(np.stack(negs, 0)).long().to(device)

        return sources, destinations, midways, negs
    
    @staticmethod
    def prepare_robustDAA_batch(batch_meta, device):
        src_arrs, tgt_arrs = batch_meta[0], batch_meta[1]
        src_arrs = torch.as_tensor(src_arrs, dtype=torch.float32, device=device)
        tgt_arrs = torch.as_tensor(tgt_arrs, dtype=torch.float32, device=device)
        
        # 预训练阶段，len(batch_meta)=4, 包含由src_arrs分出的L_D, S
        # 下游任务中，len(batch_meta)=2, L_D, S实际分别为src_arrs, tgt_arrs
        L_D, S = batch_meta[-2], batch_meta[-1]
        L_D = torch.as_tensor(L_D, dtype=torch.float32, device=device)
        S = torch.as_tensor(S, dtype=torch.float32, device=device)
        return src_arrs, tgt_arrs, L_D, S 


def cal_model_size(model):
    """ Calculate the total size (in megabytes) of a torch module. """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def geo_distance(lng1, lat1, lng2, lat2):
    """ Calculcate the geographical distance between two points (or one target point and an array of points). """
    lng1, lat1, lng2, lat2 = map(np.radians, [lng1, lat1, lng2, lat2])
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    distance = 2 * np.arcsin(np.sqrt(a)) * 6371 * 1000
    return distance


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value


class MinMaxScaler():
    """
    MinMax the input
    """
    def __init__(self, max, min):
        self.max = max
        self.min = min

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min

class StandardScaler():
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class TransferFunction():
    def __init__(self, args):
        self.args = args

    @staticmethod
    def lonlat2meters(lon, lat):
        '''经纬度转米'''
        semimajoraxis = 6378137.0
        east = lon * 0.017453292519943295
        north = lat * 0.017453292519943295
        t = math.sin(north)
        x = semimajoraxis * east
        y = 3189068.5 * math.log((1 + t) / (1 - t))
        return x, y

    @staticmethod
    def meters2lonlat(x, y):
        '''米转经纬度'''
        semimajoraxis = 6378137.0
        lon = x / semimajoraxis / 0.017453292519943295
        t = math.exp(y / 3189068.5)
        lat = math.asin((t - 1) / (t + 1)) / 0.017453292519943295
        return lon, lat

    def lonlat2metersOffset(self, lon, lat):
        x, y = self.lonlat2meters(lon, lat)
        xoffset = round(x - self.args.minxy[0], 6) / self.args.xstep
        yoffset = round(y - self.args.minxy[1], 6) / self.args.ystep
        return xoffset, yoffset

    def coord2cellId(self, x, y):
        xoffset = round(x - self.args.minxy[0], 6) / self.args.xstep
        yoffset = round(y - self.args.minxy[1], 6) / self.args.ystep
        xoffset = int(xoffset)
        yoffset = int(yoffset)
        return yoffset * self.args.numx + xoffset

    def cellId2coord(self, cellId):
        yoffset = cellId // self.args.numx
        xoffset = cellId % self.args.numx
        y = self.args.minxy[1] + (yoffset + 0.5) * self.args.ystep
        x = self.args.minxy[0] + (xoffset + 0.5) * self.args.xstep
        return x, y

    def lonlat2cellId(self, lon, lat):
        x,y = self.lonlat2meters(lon,lat)
        cellId = self.coord2cellId(x, y)
        return cellId

    def cellId2lonlat(self, cellId):
        x, y = self.cellId2coord(cellId)
        lon, lat = self.meters2lonlat(x, y)
        return lon, lat

    def cellId2vocab(self, cellId, hotcell2vocab, hotcell_kdtree, hotcellList):
        """
        Return the vocab id for a cell in the region.
        If the cell is not hot cell, the function will first search its nearest
        hotcell and return the corresponding vocab id.
        """
        if cellId in hotcell2vocab:
            return hotcell2vocab[cellId]
        else: # the cell is not hotcell, search its nearest hotcell
            hotcellId, _ = self.knearestHotcells(cellId, k=1,
                                                 hotcell_kdtree=hotcell_kdtree,
                                                 hotcellList=hotcellList)
            return hotcell2vocab[hotcellId]

    def lonlat2vocab(self, lon, lat, hotcell2vocab, hotcell_kdtree, hotcellList):
        if not self.if_inRegion(lon, lat):
            return self.args.UNK
        cellId = self.lonlat2cellId(lon, lat)
        return self.cellId2vocab(cellId, hotcell2vocab, hotcell_kdtree, hotcellList)

    # -------------------------------------------------------------------------------------------------------------
    def lonlat2xyoffset(self, lon, lat):
        '''
        将经纬度转换为xy轴上的相对偏移量（经纬度最小值位置为原点）, 映射到平面图上
        例如：(116.3, 40.0)->(4,8)
        '''
        xoffset = round((lon - self.args.lons[0]) / self.args.scale)
        yoffset = round((lat - self.args.lats[0]) / self.args.scale)
        return int(xoffset), int(yoffset)

    def xyoffset2lonlat(self, xoffset, yoffset):
        ''' 相对偏移量转换为经纬度  (4,8)-> (116.3, 40.0)'''
        lon = self.args.lons[0]+xoffset*self.args.scale
        lat = self.args.lats[0]+yoffset*self.args.scale
        return lon,lat

    def offset2spaceId(self, xoffset, yoffset):
        ''' (xoffset,yoffset) -> space_cell_id  (4,8)->116'''
        return int(yoffset * self.args.numx + xoffset)

    def spaceId2offset(self, space_cell_id):
        ''' space_cell_id -->(x,y) 116->(4.8)'''
        yoffset = space_cell_id // self.args.numx
        xoffset = space_cell_id % self.args.numx
        return int(xoffset), int(yoffset)

    def gps2spaceId(self, lon, lat):
        ''' gps--> space_cell_id  116.3,40->116'''
        xoffset, yoffset = self.lonlat2xyoffset(lon, lat)
        space_cell_id = self.offset2spaceId(xoffset, yoffset)
        return int(space_cell_id)

    def spaceId2gps(self, space_cell_id):
        '''space_cell_id -->gps 116->116.3,40'''
        xoffset, yoffset = self.spaceId2offset(space_cell_id)
        lon,lat = self.xyoffset2lonlat(xoffset,yoffset)
        return lon,lat

    def spaceId2mapId(self, space_id, t):
        ''' space_cell_id+t --> map_id  116,10->1796'''
        return int(space_id + t*self.args.space_cell_size)

    def mapId2spaceId(self, map_id):
        ''' map_id -->space_cell_id  1796-> 116'''
        return int(map_id % self.args.space_cell_size)

    def mapId2t(self, map_id):
        ''' map_id -->t 1796-> 10'''
        return int(map_id // self.args.space_cell_size)

    def mapId2word(self, map_id, hotcell2word, tree, hot_ts):
        ''' map_id -> vocal_id 若不在热度词中，则用与其较近的词代替 '''
        word = hotcell2word.get(map_id, self.args.UNK)
        return word if word != self.args.UNK else self.get_a_nn(map_id, tree, hot_ts)

    @staticmethod
    def word2mapId(word, word2hotcell):
        ''' word -> map_id 不会存在查找不到的情况'''
        return word2hotcell.get(word, 0)

    def word2xyt(self, word, word2hotcell):
        ''' word --> xoffset,yoffset,t '''
        map_id = self.word2mapId(word, word2hotcell)
        t = self.mapId2t(map_id)
        space_id = self.mapId2spaceId(map_id)
        xoffset,yoffset = self.spaceId2offset(space_id)
        return xoffset,yoffset,t

    def xyt2word(self,x,y,t, hotcell2word, tree, hot_ts):
        ''' xoffset,yoffset,t -->word '''
        space_id = self.offset2spaceId(x, y)
        map_id = self.spaceId2mapId(space_id, t)
        word = self.mapId2word(map_id, hotcell2word, tree, hot_ts)
        return word

    def mapIds2words(self, map_ids, hotcell2word, tree, hot_ts):
        ''' map_ids --> words'''
        words = [self.mapId2word(id, hotcell2word, tree, hot_ts) for id in map_ids]
        return words

    @staticmethod
    def words2mapIds(words, word2hotcell):
        '''words -+--> map_ids'''
        map_ids = [word2hotcell.get(id, 0) for id in words]
        return map_ids

    # -------------------------------------------------------------------------------------------------------------
    def trip2spaceIds(self, trip):
        ''' trip --> space_cell_ids '''
        space_ids = []
        for lonlat in trip:
            space_id = self.gps2spaceId(lonlat[0], lonlat[1])
            space_ids.append(space_id)
        return space_ids

    def trip2mapIDs(self, trip, ts):
        ''' trip --> space_cell_ids --> map_ids'''
        map_ids = []
        for (lon, lat), t in zip(trip, ts):
            space_id = self.gps2spaceId(lon, lat)
            t = int(t) // self.args.time_span
            map_id = self.spaceId2mapId(space_id, t)
            map_ids.append(map_id)
        return list(map_ids)

    def trip2words(self, trip, ts, hotcell2word, tree, hot_ts):
        ''' 减少迭代次数的trip2words '''
        words = []
        for (lon, lat), t in zip(trip, ts):
            space_id = self.gps2spaceId(lon, lat)
            t = int(t) // self.args.time_span
            map_id = self.spaceId2mapId(space_id, t)
            words.append(self.mapId2word(map_id, hotcell2word, tree, hot_ts))
        return words

    def trip2seq(self, trip, hotcell2vocab, hotcell_kdtree, hotcellList):
        seq = []
        for i in range(trip.shape[0]):
            lon, lat = trip[i, :]
            vocab = self.lonlat2vocab(lon, lat, hotcell2vocab, hotcell_kdtree, hotcellList)
            if len(seq) != 0 and vocab == seq[-1]:
                continue
            else:
                seq.append(vocab)
        return seq

    # -------------------------------------------------------------------------------------------------------------
    @staticmethod
    def tripmeta(trip, ts):
        ''' 得到一个trip的meta 经纬度+时间重心坐标'''
        long_min = min([p[0] for p in trip])
        long_max = max([p[0] for p in trip])
        lat_min = min([p[1] for p in trip])
        lat_max = max([p[1] for p in trip])
        min_ts = min(ts)
        max_ts = max(ts)
        ts_centroids = min_ts+(max_ts-min_ts)/2

        long_centroids = long_min + (long_max - long_min) / 2
        lat_centroids = lat_min + (lat_max - lat_min) / 2
        return round(long_centroids, 8), round(lat_centroids, 8), round(ts_centroids, 4)

    def get_tripmeta(self, trip):
        # Calculate centroid for longitude and latitude
        long_min = min([p[0] for p in trip])
        long_max = max([p[0] for p in trip])
        lat_min = min([p[1] for p in trip])
        lat_max = max([p[1] for p in trip])
        long_centroids = long_min + (long_max - long_min) / 2
        lat_centroids = lat_min + (lat_max - lat_min) / 2

        # Convert centroid coordinates to offsets
        xs, ys = self.lonlat2metersOffset(long_centroids, lat_centroids)

        return xs / self.args.xstep, ys / self.args.ystep

    # -------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_snn(long, lat, tree, K):
        """
        获取一个 经纬度点 的 空间最近词

        :param long: 经度
        :param lat: 纬度
        :param tree: ckdtree
        :param K: top K
        :return:
        """
        point = np.array([long,lat])
        k_dists, k_indexs = tree.query(point, K, p=1)
        return k_dists, k_indexs+4

    def get_a_nn(self, map_id, tree, hot_ts):
        """
        获得一个离 mapId最近的 hot word
        2021/11/20 检查正确性完毕

        :param map_id: 映射到的时空编码值
        :param tree: ckdtree
        :param hot_ts: 用于在空间邻居的基础上继续查找时间上的邻居
        :return: 与mapId最近的 hot word，也是一个时空编码值
        """
        # 得到非热度词的经纬度和时间
        space_id = self.mapId2spaceId(map_id)
        lon, lat = self.spaceId2gps(space_id)
        t = self.mapId2t(map_id)
        # 得到在热度词中的空间邻居, 为+4后的编码值
        k_dists, k_indexs = self.get_snn(lon, lat, tree, self.args.space_nn_topK)
        # k_dists, k_indexs = k_dists.tolist(), k_indexs.tolist()
        # 选择时间最接近的热度词输出
        # 初始化一个最小的时间差距 和 一个最小邻居
        min_hot_t = 1000
        nn = self.args.UNK
        # 经验证以下找最近邻比利用array快
        for hot_word in k_indexs:
            # 注意这里需要-4来获取每个空间邻居所处的时间段
            if abs(hot_ts[hot_word-self.args.start]-t) < min_hot_t:
                min_hot_t = abs(hot_ts[hot_word-self.args.start]-t)
                nn = hot_word
        return nn

    def if_inRegion(self, lon, lat):
        return lon >= self.args.lons[0] and lon < self.args.lons[1] and lat >= self.args.lats[0] and lat < self.args.lats[1]

    def knearestHotcells(self, cellId, k, hotcell_kdtree, hotcellList):
        coord = self.cellId2coord(cellId)
        dists, idxs = hotcell_kdtree.query(coord, k)
        return np.array(hotcellList)[idxs], dists


def pad_arrays_pair(src, trg, keep_invp=False):
    """
    1. 对轨迹补零操作，使得所有轨迹的长度都一样长
    2. 对轨迹长度从大到小进行排序
    3. 返回TD类，每代表一个轨迹点
    4. 返回形式 ['src', 'lengths', 'trg', 'invp']

    :param src: (list[array[int32]])
    :param trg: (list[array[int32]])
    :param keep_invp: 是否需要保留原来的轨迹长度排序索引
    :return:

    src (batch, seq_len1)
    trg (batch, seq_len2)
    lengths (batch,)
    invp (batch,): inverse permutation, src.t()[invp] gets original order
    """
    TD = namedtuple('TD', ['src', 'lengths', 'trg', 'invp'])
    assert len(src) == len(trg), "source and target should have the same length"
    idx = argsort(src)
    src = list(np.array(src)[idx])
    trg = list(np.array(trg)[idx])

    lengths = list(map(len, src))
    lengths = torch.LongTensor(lengths)
    src = pad_arrays(src)
    trg = pad_arrays(trg)
    if keep_invp == True:
        invp = torch.LongTensor(invpermute(idx))
        # return TD(src=src.t().contiguous(), lengths=lengths.view(1, -1), trg=trg.t().contiguous(), invp=invp)
        return TD(src=src.contiguous(), lengths=lengths.view(-1), trg=trg.contiguous(), invp=invp)
    else:
        # return TD(src=src.t().contiguous(), lengths=lengths.view(1, -1), trg=trg.t().contiguous(), invp=[])
        return TD(src=src.contiguous(), lengths=lengths.view(-1), trg=trg.contiguous(), invp=[])

def pad_arrays(a):
    """
    多条轨迹(一个批次的轨迹)补位操作，每条轨迹的长度补0，使其长度和该batch最长轨迹的长度相同
    :param a: 一批轨迹
    :return:
    """
    max_length = max(map(len, a))
    a = [pad_array(a[i], max_length) for i in range(len(a))]
    a = np.stack(a).astype(np.int)
    return torch.LongTensor(a)

def pad_array(a, max_length, PAD=0):
    """
    :param a: 一条待补位操作的轨迹 (array[int32])
    :param max_length: 该条轨迹所在批次轨迹中 轨迹的最大长度，按该长度标准补位
    :param PAD: 补位值，设置为0
    :return: 补位后的轨迹
    """
    return np.concatenate((a, [PAD]*(max_length - len(a))))


def argsort(seq):
    """
    sort by length in reverse order
    ex. src=[[1,2,3],[3,4,5,6],[2,3,4,56,3]] ，return 2，1，0
    :param seq: (list[array[int32]])
    :return: the reversed order
    """
    return [x for x, y in sorted(enumerate(seq), key=lambda x: len(x[1]), reverse=True)]

def invpermute(p):
    """
    输入p,返回p的每个位置的值的索引invp
    idx = [5, 7, 8, 9, 6, 1, 2, 0, 3, 4]
    invp(idx) = [7, 5, 6, 8, 9, 0, 4, 1, 2, 3]
    invp[p[i]] = i 如 p中有个数是45，我现在想知道45在p的第几个位置，那么invp[45]会告诉我们答案
    invp[i] = p.index(i)

    inverse permutation
    """
    p = np.asarray(p)
    invp = np.empty_like(p)
    for i in range(p.size):
        invp[p[i]] = i
    return invp

def random_subseq(a, rate):
    """
    Dropping some points between a[1:-2] randomly according to rate.

    Input:
    a (array[int])
    rate (float)
    """
    idx = np.random.rand(len(a)) < rate
    idx[0], idx[-1] = True, True
    return a[idx]


def k_shortest_paths(G, source, target, k, weight=None):
    return list(
        islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
    )


# def k_shortest_paths(G, source, target, k=1, weight='weight'):
#     """Returns the k-shortest paths from source to target in a weighted graph G.
#
#     Parameters
#     ----------
#     G : NetworkX graph
#
#     source : node
#        Starting node
#
#     target : node
#        Ending node
#
#     k : integer, optional (default=1)
#         The number of shortest paths to find
#
#     weight: string, optional (default='weight')
#        Edge data key corresponding to the edge weight
#
#     Returns
#     -------
#     lengths, paths : lists
#        Returns a tuple with two lists.
#        The first list stores the length of each k-shortest path.
#        The second list stores each k-shortest path.
#
#     Raises
#     ------
#     NetworkXNoPath
#        If no path exists between source and target.
#
#     Examples
#     --------
#     >>> G=nx.complete_graph(5)
#     >>> print(k_shortest_paths(G, 0, 4, 4))
#     ([1, 2, 2, 2], [[0, 4], [0, 1, 4], [0, 2, 4], [0, 3, 4]])
#
#     Notes
#     ------
#     Edge weight attributes must be numerical and non-negative.
#     Distances are calculated as sums of weighted edges traversed.
#
#     """
#     if source == target:
#         return ([0], [[source]])
#
#     try:
#         length, path = nx.single_source_dijkstra(G, source, target, weight=weight)
#     except nx.NetworkXNoPath:
#         raise nx.NetworkXNoPath("node %s not reachable from %s" % (source, target))
#
#     lengths = [length]
#     paths = [path]
#     c = count()
#     B = []
#     G_original = G.copy()
#
#     for i in range(1, k):
#         for j in range(len(paths[-1]) - 1):
#             spur_node = paths[-1][j]
#             root_path = paths[-1][:j + 1]
#
#             edges_removed = []
#             for c_path in paths:
#                 if len(c_path) > j and root_path == c_path[:j + 1]:
#                     u = c_path[j]
#                     v = c_path[j + 1]
#                     if G.has_edge(u, v):
#                         edge_attr = G.edges[u, v]
#                         G.remove_edge(u, v)
#                         edges_removed.append((u, v, edge_attr))
#
#             for n in range(len(root_path) - 1):
#                 node = root_path[n]
#                 tmp_removed = []
#                 # out-edges
#                 for u, v, edge_attr in G.edges(node, data=True):
#                     tmp_removed.append((u, v))
#                     edges_removed.append((u, v, edge_attr))
#
#                 if G.is_directed():
#                     # in-edges
#                     for u, v, edge_attr in G.in_edges(node, data=True):
#                         tmp_removed.append((u, v))
#                         edges_removed.append((u, v, edge_attr))
#
#                 for u, v in tmp_removed:
#                     G.remove_edge(u, v)
#
#             try:
#                 spur_path_length, spur_path = nx.single_source_dijkstra(G, spur_node, target, weight=weight)
#             except nx.NetworkXNoPath:
#                 raise nx.NetworkXNoPath("node %s not reachable from %s" % (spur_node, target))
#
#             total_path = root_path[:-1] + spur_path
#             total_path_length = get_path_length(G_original, root_path, weight) + spur_path_length
#             heappush(B, (total_path_length, next(c), total_path))
#
#             for e in edges_removed:
#                 u, v, edge_attr = e
#                 G.add_edge(u, v, **edge_attr)
#
#         if B:
#             (l, _, p) = heappop(B)
#             lengths.append(l)
#             paths.append(p)
#         else:
#             break
#
#     return (lengths, paths)
#
#
# def get_path_length(G, path, weight='weight'):
#     length = 0
#     if len(path) > 1:
#         for i in range(len(path) - 1):
#             u = path[i]
#             v = path[i + 1]
#
#             length += G.edges[u, v].get(weight, 1)
#
#     return length



class SDMSampleGenerator:
    """
    Source-Destination-Midway Word2Vec-like sample generator
    """
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, data, min_count):

        self.negatives = []
        self.discards = []
        self.negpos = 0

        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()

        self.read_words(data, min_count)
        self.initTableNegatives()
        self.initTableDiscards()

    def read_words(self, data, min_count):
        word_frequency = dict()
        for slice in data:
            if len(slice) > 1:
                self.sentences_count += 1
                for segment in slice:
                    self.token_count += 1
                    word_frequency[segment] = word_frequency.get(segment, 0) + 1

                    if self.token_count % 1000000 == 0:
                        print("Read " + str(int(self.token_count / 1000000)) + "M segments.")

        wid = 0
        for w, c in word_frequency.items():
            if c < min_count:
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        print("Total embeddings: " + str(len(self.word2id)))

    def initTableDiscards(self):
        t = 0.0001
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)

    def initTableNegatives(self):
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.5
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * SDMSampleGenerator.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)

    def getNegatives(self, target, size):  # TODO check equality with target
        response = self.negatives[self.negpos:self.negpos + size]
        self.negpos = (self.negpos + size) % len(self.negatives)
        if len(response) != size:
            return np.concatenate((response, self.negatives[0:self.negpos]))
        return response


def clip(n, minValue, maxValue):
    return min(max(n, minValue), maxValue)


def map_size(levelOfDetail):
    return 256 << levelOfDetail


def latlon2pxy(latitude, longitude, level_of_detail, min_latitude, max_latitude, min_longitude, max_longitude):
    latitude = clip(latitude, min_latitude, max_latitude)
    longitude = clip(longitude, min_longitude, max_longitude)

    x = (longitude + 180) / 360
    sinLatitude = math.sin(latitude * math.pi / 180)
    y = 0.5 - math.log((1 + sinLatitude) / (1 - sinLatitude)) / (4 * math.pi)

    mapSize = map_size(level_of_detail)
    pixelX = int(clip(x * mapSize + 0.5, 0, mapSize - 1))
    pixelY = int(clip(y * mapSize + 0.5, 0, mapSize - 1))
    return pixelX, pixelY


def pxy2txy(pixelX, pixelY):
    tileX = pixelX // 256
    tileY = pixelY // 256
    return tileX, tileY


def txy2quadkey(tileX, tileY, levelOfDetail):
    quadKey = []
    for i in range(levelOfDetail, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (tileX & mask) != 0:
            digit += 1
        if (tileY & mask) != 0:
            digit += 2
        quadKey.append(str(digit))

    return ''.join(quadKey)


def latlon2quadkey(lat, lon, level, min_lat, max_lat, min_lng, max_lng):
    pixelX, pixelY = latlon2pxy(lat, lon, level, min_lat, max_lat, min_lng, max_lng)
    tileX, tileY = pxy2txy(pixelX, pixelY)
    return txy2quadkey(tileX, tileY, level)


def torch_delete(arr: torch.Tensor, ind: list, dim: int) -> torch.Tensor:
    skip = [i for i in range(arr.size(dim)) if i not in ind]
    indices = [slice(None) if i != dim else skip for i in range(arr.ndim)]
    return arr.__getitem__(indices)
