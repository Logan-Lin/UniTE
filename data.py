from argparse import ArgumentParser
import copy
import os
import math
import random
from collections import Counter, defaultdict
from itertools import islice, zip_longest
from time import time
import platform
import json
import gc
import h5py
from copy import deepcopy
from itertools import combinations

import torch
from torch import nn
import pandas as pd
import numpy as np
from einops import repeat, rearrange
from scipy import sparse, spatial
from tqdm import tqdm, trange
import networkx as nx
from sklearn.utils import shuffle
from sklearn.neighbors import NearestNeighbors, BallTree

from utils import create_if_noexists, remove_if_exists, intersection, geo_distance, idDocker, DotDict, TransferFunction, next_batch
from utils import k_shortest_paths, latlon2quadkey

pd.options.mode.chained_assignment = None
CLASS_COL = 'driver'
SET_NAMES = [(0, 'train'), (1, 'val'), (2, 'test')]
MIN_TRIP_LEN = 6
MAX_TRIP_LEN = 120
TARGET_SAMPLE_RATE = 15
TRIP_COLS = ['tod', 'road', 'road_prop', 'lng', 'lat', 'weekday', 'seq_i', 'seconds']

DATASET_PATH = os.environ['DATASET_PATH']
META_PATH = os.environ['META_PATH']


class Data:
    def __init__(self, name, road_type='road_network'):
        self.name = name

        self.base_path = META_PATH
        self.dataset_path = DATASET_PATH

        self.df_path = f'{self.dataset_path}/{self.name}.h5'
        self.meta_dir = f'{self.base_path}/meta/{self.name}'
        self.stat_path = f'{self.meta_dir}/stat_grid.h5' if road_type == 'grid' else f'{self.meta_dir}/stat.h5'

        self.get_meta_path = lambda meta_type, select_set: os.path.join(
            self.meta_dir, f'{meta_type}_' + \
                           ('grid_' if road_type == 'grid' else '') + \
                           f'{select_set}.npz'
        )

        assert road_type in ['road_network', 'grid'], "road type must be road_network or grid"
        self.road_type = road_type

    """ Load functions for loading dataframes and meta. """

    def read_hdf(self):
        # Load the raw data from HDF files.
        # One set of raw dataset is composed of one HDF file with four keys.
        # The trips contains the sequences of trajectories, with three columns: trip, time, road
        self.trips = pd.read_hdf(self.df_path, key='trips')
        # The trip_info contains meta information about trips. For now, this is mainly used for class labels.
        self.trip_info = pd.read_hdf(self.df_path, key='trip_info')
        # The road_info contains meta information about roads.
        self.road_info = pd.read_hdf(self.df_path, key='road_info')
        # self.trips = pd.merge(self.trips, self.road_info[['road', 'lng', 'lat']], on='road', how='left')
        self.network_info = None
        self.network = None

        if self.road_type == 'grid':
            self.project_to_grid()

        # Add some columns to the trip
        self.trips['seconds'] = self.trips['time'].apply(lambda x: x.timestamp())
        self.trips['tod'] = self.trips['seconds'] % (24 * 60 * 60) / (24 * 60 * 60)
        self.trips['weekday'] = self.trips['time'].dt.weekday
        self.stat = self.trips.describe()

        num_road = int(self.road_info['road'].max() + 1)
        num_class = int(self.trip_info[CLASS_COL].max() + 1)
        if self.road_type == 'grid':
            self.data_info = pd.Series([num_road, num_class, self.num_w, self.num_h],
                                       index=['num_road', 'num_class', 'num_w', 'num_h'])
        else:
            self.data_info = pd.Series([num_road, num_class], index=['num_road', 'num_class'])
        print('Loaded DataFrame from', self.df_path)
        self.num_road = num_road

        num_trips = self.trip_info.shape[0]
        self.train_val_test_trips = (self.trip_info['trip'].iloc[:int(num_trips * 0.8)],
                                     self.trip_info['trip'].iloc[int(num_trips * 0.8):int(num_trips * 0.9)],
                                     self.trip_info['trip'].iloc[int(num_trips * 0.9):])

        create_if_noexists(self.meta_dir)
        self.stat.to_hdf(self.stat_path, key='stat')
        self.data_info.to_hdf(self.stat_path, key='info')
        print(self.stat)
        print(self.data_info)
        print('Dumped dataset info into', self.stat_path)

        self.valid_trips = [self.get_valid_trip_id(i) for i in range(3)]

    def project_to_grid(self):
        print("Project trips to grids, ", end='')
        delta_degree = 0.0015
        min_lng, max_lng = self.trips['lng'].min(), self.trips['lng'].max()
        min_lat, max_lat = self.trips['lat'].min(), self.trips['lat'].max()
        num_h = math.ceil((max_lat - min_lat) / delta_degree)
        num_w = math.ceil((max_lng - min_lng) / delta_degree)
        self.num_h = num_h
        self.num_w = num_w
        self.num_road = num_w * num_h

        def _project_lng_to_w(lng):
            return np.clip(np.ceil((lng - min_lng) / (max_lng - min_lng) * num_w) - 1, 0,
                           num_w - 1)

        def _project_lat_to_h(lat):
            return np.clip(np.ceil((lat - min_lat) / (max_lat - min_lat) * num_h) - 1, 0,
                           num_h - 1)

        ws = self.trips['lng'].apply(_project_lng_to_w)
        hs = self.trips['lat'].apply(_project_lat_to_h)
        self.trips['road'] = [h * num_w + w for h, w in zip(hs, ws)]

        road_lngs = np.arange(num_w) * delta_degree + delta_degree / 2 + min_lng
        road_lats = np.arange(num_h) * delta_degree + delta_degree / 2 + min_lat
        road_lngs = repeat(road_lngs, 'W -> (H W)', H=num_h)
        road_lats = repeat(road_lats, 'H -> (H W)', W=num_w)

        self.road_info = pd.DataFrame({
            "road": list(range(num_w * num_h)),
            "lng": road_lngs,
            "lat": road_lats
        })

    def load_stat(self):
        # Load statistical information for features.
        self.stat = pd.read_hdf(self.stat_path, key='stat')
        self.data_info = pd.read_hdf(self.stat_path, key='info')

    def load_meta(self, meta_type, select_set):
        meta_path = self.get_meta_path(meta_type, select_set)
        loaded = np.load(meta_path, allow_pickle=True)
        print('Loaded meta from', meta_path)
        return list(loaded.values())

    def build_network(self):
        if self.network is None:
            network = nx.Graph()
            nodes = list(set(self.road_info['o'].unique()) | set(self.road_info['d'].unique()))
            network.add_nodes_from(nodes)
            # network.add_edges_from(self.network_info[['o', 'd']].to_numpy().astype(int),
            #                        distance=self.network_info['length'].to_numpy().astype(float))
            network.add_weighted_edges_from(self.road_info[['o', 'd', 'length']].to_numpy(), weight='length')
            self.network = network

    def get_valid_trip_id(self, select_set):
        select_trip_id = self.train_val_test_trips[select_set]
        trips = self.trips[self.trips['trip'].isin(select_trip_id)]
        valid_trip_id = []
        for _, group in tqdm(trips.groupby('trip'), desc='Filtering trips', total=select_trip_id.shape[0], leave=False):
            if (not group.isna().any().any()) and group.shape[0] >= MIN_TRIP_LEN and group.shape[0] <= MAX_TRIP_LEN:
                if ((group['seconds'] - group.shift(1)['seconds']).iloc[1:] == TARGET_SAMPLE_RATE).all():
                    valid_trip_id.append(group.iloc[0]['trip'])
        return valid_trip_id

    def dump_meta(self, meta_type, select_set):
        """
        Dump meta data into numpy binary files for fast loading later.

        :param meta_type: type name of meta data to dump.
        :param select_set: index of set to dump. 0 - training set, 1 - validation set, and 2 - testing set.
        """
        # Prepare some common objects that will probably be useful for various types of meta data.
        select_trip_id = self.valid_trips[select_set]
        known_trips = self.trips[self.trips['trip'].isin(self.valid_trips[0] + self.valid_trips[1])]
        trips = self.trips[self.trips['trip'].isin(select_trip_id)]
        trip_info = self.trip_info[self.trip_info['trip'].isin(select_trip_id)]
        max_trip_len = max(Counter(trips['trip']).values())
        trip_normalizer = Normalizer(self.stat, feat_cols=[0, 2, 3, 4], norm_type='minmax')

        if meta_type == 'trip':
            """
            The "trip" meta data obeys the original form of trajectories. 
            One complete trajectory sequence is regarded as one trip.
            """
            arrs, valid_lens = [], []
            for _, group in tqdm(trips.groupby('trip'), desc='Gathering trips', total=len(select_trip_id)):
                arr = group[TRIP_COLS].to_numpy()
                valid_len = arr.shape[0]

                offset = group['time'].apply(lambda x: x.timestamp()).to_numpy()
                offset = (offset - offset[0]) / (offset[-1] - offset[0]) * 2 - 1
                arr = np.append(arr, offset.reshape(-1, 1), 1)

                # Pad all trips to the maximum length by repeating the last item.
                arr = np.concatenate([arr, np.repeat(arr[-1:], max_trip_len - valid_len, axis=0)], 0)
                arrs.append(arr)
                valid_lens.append(valid_len)
            # Features of arrs: TRIP_COLS + [offset]
            arrs, valid_lens = np.stack(arrs, 0), np.array(valid_lens)
            arrs = trip_normalizer(arrs)

            meta = [arrs, valid_lens]

        elif 'timemat' in meta_type:
            """
            Construct time matrix of meta types
            """
            params = meta_type.split('-')
            params.remove('timemat')
            sup_meta_type = '-'.join(params)

            try:
                trip_arrs, valid_lens = self.load_meta(sup_meta_type, select_set)[:2]
            except FileNotFoundError:
                raise FileNotFoundError("Dump meta type {} first.".format(sup_meta_type))

            time_arrs = trip_arrs[:, :, 7]  # seconds
            meta = [gen_time_mat(time_arrs), ]

        elif 'traj2vectime' in meta_type:
            params = meta_type.split('-')
            params.remove('traj2vectime')
            sup_meta_type = '-'.join(params)

            try:
                trip_arrs, valid_lens = self.load_meta(sup_meta_type, select_set)
            except FileNotFoundError:
                raise FileNotFoundError("Dump meta type {} first.".format(sup_meta_type))

            road_list = trip_arrs[:, :, 1].astype(int)
            time_arrs = trip_arrs[:, :, 7]  # seconds
            time_diff = time_arrs[:, 1:] - time_arrs[:, :-1]
            time_diff = np.concatenate([np.zeros([len(time_arrs), 1]), time_diff], axis=1)
            meta = [road_list, valid_lens, time_diff]

        elif meta_type == 'class' or meta_type == 'classbatch':
            """
            The "class" meta data provides classification label.
            Typically, the label calculates from user or driver ID.
            """
            classes = trip_info[CLASS_COL].to_numpy()
            meta = [classes]

        elif meta_type == 'tte':
            """
            The "tte" meta data provides travel time estimation label.
            The ground truth travel time is the time span of trajectories in minutes.
            """
            travel_times = []
            for _, row in tqdm(trip_info.iterrows(), desc='Gathering TTEs', total=trip_info.shape[0]):
                travel_times.append((row['end'] - row['start']).total_seconds() / 60)
            travel_times = np.array(travel_times)

            meta = [travel_times]

        elif 'resample' in meta_type:
            """
            The "resample" meta data records the resampled version of "trip" meta data.
            The trajectories is resampled along time axis given a certain sampling rate.
            The last point of trajectories is always appended.
            """
            # Parameters are given in the meta_type. Use format "resample-{sample_rate}"
            params = meta_type.split('-')
            # Sample rate (in minutes) of the sparse trajectories.
            sample_rate = float(params[1])

            # Resample trips, also gather road segment candidates based on spatial distance.
            arrs, lengths = [], []
            for _, group in tqdm(trips.groupby('trip'), desc='Resampling trips', total=len(select_trip_id)):
                reindex_trip = group.reset_index().set_index('time')
                resampled_trip = reindex_trip[~reindex_trip.index.duplicated()].resample(
                    rule=pd.Timedelta(sample_rate, 'seconds'), origin='start').asfreq()
                resampled_trip = resampled_trip[resampled_trip['seq_i'].notnull()]  # resample would result in Nan
                resampled_arr = resampled_trip[TRIP_COLS].to_numpy()
                # Append the last point of this trajectory if it is absent in the resampled trajectory.
                if resampled_trip['seq_i'].iloc[-1] != group['seq_i'].iloc[-1]:
                    resampled_arr = np.append(resampled_arr, group.iloc[-1:][TRIP_COLS].to_numpy(), 0)

                arrs.append(resampled_arr)
                lengths.append(resampled_arr.shape[0])

            lengths = np.array(lengths)

            # Pad all resampled trips to the maximum length.
            max_resampled_length = lengths.max()
            # Features of resampled trips: TRIP_COL
            arrs = np.stack([
                np.concatenate([trip, np.repeat(trip[-1:], max_resampled_length - trip.shape[0], 0)], 0)
                for trip in arrs], 0)
            arrs = trip_normalizer(arrs)

            meta = [arrs, lengths]

        elif 'distcand' in meta_type:
            """
            The "distcand" meta data records the distance candidates given a certain distance threshold in meters.
            The road segments within the distance threshold of GPS points are recorded.
            """
            # Parameters are given in the meta_type. Use format "distcand-{dist_thres}".
            dist_thres = float(meta_type.split('-')[1])

            # Prepare the coordinates of roads.
            road_coors = self.road_info[['road', 'road_lng', 'road_lat']].to_numpy()
            cand_seq = []
            max_num_cand = 0
            for _, group in tqdm(trips.groupby('trip'), desc='Finding distance candidates', total=len(select_trip_id)):
                cand_row = []
                for _, row in group.iterrows():
                    # Calculate the geographical distance between this GPS point and all road segments.
                    dist = geo_distance(row['lng'], row['lat'], road_coors[:, 1], road_coors[:, 2])
                    cand = road_coors[:, 0][dist <= dist_thres].astype(int).tolist()
                    max_num_cand = max(len(cand), max_num_cand)
                    cand_row.append(cand)
                cand_seq.append(cand_row)

            pad_seq = []
            for cand_row in cand_seq:
                cand_row = [cand + [-1] * (max_num_cand - len(cand)) for cand in cand_row]
                pad_seq.append(cand_row + [cand_row[-1]] * (max_trip_len - len(cand_row)))
            pad_seq = np.array(pad_seq).astype(int)

            meta = [pad_seq]

        elif 'knncand' in meta_type:
            """
            The "knncand" meta data records the k-nearest neighbor candidates given a certain k value.
            The road segments that are within the set of kNN of GPS points are recorded.
            """
            # Parameters are given in the meta_type. Use format "knncand-{k}".
            k = int(meta_type.split('-')[1])

            # Prepare the coordinates of roads and the knn model.
            road_coors = self.road_info[['road', 'road_lng', 'road_lat']].to_numpy()
            knn = NearestNeighbors(n_neighbors=k)
            knn.fit(road_coors[:, [1, 2]])
            cand_seq = []
            for _, group in tqdm(trips.groupby('trip'), desc='Finding knn candidates', total=len(select_trip_id)):
                _, neighbors = knn.kneighbors(group[['lng', 'lat']].to_numpy())
                cand_seq.append(neighbors)

            pad_seq = []
            for cand_row in cand_seq:
                pad_seq.append(np.concatenate([cand_row, np.repeat(
                    cand_row[-1:], max_trip_len - cand_row.shape[0], 0)], 0))
            pad_seq = np.stack(pad_seq, 0).astype(int)

            meta = [pad_seq]

        elif 'fromto' in meta_type:
            """
            The "fromto" meta data records the next and previous road segments of trajectories.
            """
            from_to_roads, valid_lengths = [], []
            for _, group in tqdm(trips.groupby('trip'), desc='Gathering from-to sequences', total=len(select_trip_id)):
                from_to_road = group[['road', 'seq_i']].to_numpy().astype(int)
                from_to_road = np.stack([from_to_road[1:, 0], from_to_road[1:, 1],
                                         from_to_road[:-1, 0], from_to_road[:-1, 1]], 1)
                valid_length = from_to_road.shape[0]
                from_to_road = np.append(
                    from_to_road, np.repeat(from_to_road[-1:], max_trip_len - 1 - valid_length, 0), 0)

                from_to_roads.append(from_to_road)
                valid_lengths.append(valid_length)
            # Features of from_to_roads: [to_road_segments, to_seg_i, from_road_segments, from_seg_i]
            from_to_roads = np.stack(from_to_roads, 0)
            valid_lengths = np.array(valid_lengths)

            meta = [from_to_roads, valid_lengths]

        elif 'seqcand' in meta_type:
            """
            The "seqcand" meta data record the sequential candidate mask given a certain maximum number of neighbors.
            The road segments within the possible transfer candidates of the current point are recorded.
            """
            # Parameters are given in the meta_type. Use format "seqcand-{nei_thres}".
            nei_thres = int(meta_type.split('-')[1])

            # Gather all possible transfer between road segments.
            known_trip_trans = known_trips.shift(1).join(known_trips, lsuffix='_from', rsuffix='_to')
            known_trip_trans = known_trip_trans[known_trip_trans['trip_from'] == known_trip_trans['trip_to']]
            known_trip_trans = known_trip_trans[known_trip_trans['road_from'] != known_trip_trans['road_to']]
            transfer_counter = Counter((int(row['road_from']), int(row['road_to']))
                                       for index, row in known_trip_trans.iterrows())
            transfer_pairs = pd.DataFrame([[l, r, v] for (l, r), v in transfer_counter.items()], columns=[
                'road_from', 'road_to', 'freq'])

            # Form the dicts of sequential candidates.
            transfer_tos = {index: group.sort_values('freq', ascending=False)['road_to'].tolist()
                            for index, group in transfer_pairs.groupby('road_from')}
            transfer_froms = {index: group.sort_values('freq', ascending=False)['road_from'].tolist()
                              for index, group in transfer_pairs.groupby('road_to')}

            trip_trans = trips.shift(1).join(trips, lsuffix='_from', rsuffix='_to')
            trip_trans = trip_trans[trip_trans['trip_from'] == trip_trans['trip_to']]
            to_cand_seq, from_cand_seq = [], []
            for _, group in tqdm(trip_trans.groupby('trip_from'), desc='Finding sequential candidates',
                                 total=len(select_trip_id)):
                to_cand_row, from_cand_row = [], []
                for _, row in group.iterrows():
                    # Fetch sequential candidates and calculate corresponding masks.
                    to_cand = transfer_tos.get(row['road_from'], [row['road_from']])[:nei_thres]
                    from_cand = transfer_froms.get(row['road_to'], [row['road_to']])[:nei_thres]
                    to_cand_row.append(to_cand + [-1] * (nei_thres - len(to_cand)))
                    from_cand_row.append(from_cand + [-1] * (nei_thres - len(from_cand)))
                to_cand_seq.append([[-1] * nei_thres] + to_cand_row)
                from_cand_seq.append(from_cand_row + [[-1] * nei_thres])

            pad_to_cand, pad_from_cand = [], []
            for to_cand_row, from_cand_row in zip(to_cand_seq, from_cand_seq):
                pad_to_cand.append(to_cand_row + [to_cand_row[-1]] * (max_trip_len - len(to_cand_row)))
                pad_from_cand.append(from_cand_row + [from_cand_row[-1]] * (max_trip_len - len(from_cand_row)))
            pad_to_cand = np.array(pad_to_cand).astype(int)
            pad_from_cand = np.array(pad_from_cand).astype(int)

            meta = [pad_to_cand, pad_from_cand]

        elif 'mlm' in meta_type:
            """
            Meta type corresponds to Masked Language Model-style pre-training.
            """
            # Parameters are given in the meta_type. Use format "mlm-{sample_rate}".
            params = meta_type.split('-')
            sample_rate = float(params[1])

            arrs, valid_lens = [], []
            special_token_start = self.data_info['num_road']
            for _, group in tqdm(trips.groupby('trip'), desc='Gathering MLM trips', total=len(select_trip_id)):
                reindex_trip = group.reset_index().set_index('time')
                resampled_trip = reindex_trip[~reindex_trip.index.duplicated()].resample(
                    rule=pd.Timedelta(sample_rate, 'seconds'), origin='start').asfreq()
                resampled_trip = resampled_trip[resampled_trip['seq_i'].notnull()]
                if resampled_trip['seq_i'].iloc[-1] != group['seq_i'].iloc[-1]:
                    resampled_trip = pd.concat([resampled_trip, group.iloc[[-1]]], axis=0)

                group = group.set_index('seq_i')
                group['token'] = group['road']
                group.loc[resampled_trip['seq_i'], 'token'] = special_token_start
                group = group.reset_index()

                valid_len = group.shape[0]
                # Features of MLM trips: TRIP_COL + 'token'
                arr = group[TRIP_COLS + ['token']].to_numpy()
                arr = np.concatenate([arr, np.repeat(arr[-1:], max_trip_len - valid_len, axis=0)], 0)
                arrs.append(arr)
                valid_lens.append(valid_len)

            arrs, valid_lens = np.stack(arrs, 0), np.array(valid_lens)

            meta = [arrs, valid_lens]

        elif 'trim' in meta_type or 'shift' in meta_type:
            """
            Trajectory Trimming Augmentation
            """
            # Parameters are given in the meta_type. Use format "trim-{edit_ratio}".
            params = meta_type.split('-')
            augment_type = params[0]
            edit_ratio = float(params[1])

            if augment_type == 'trim':
                data_augment_func = trim_traj
            elif augment_type == 'shift':
                data_augment_func = temporal_shift
            else:
                raise NotImplementedError("No augmentation strategy ", augment_type)

            try:
                trip_arr, valid_lens = self.load_meta('trip', select_set)
            except:
                raise ValueError("File {} does not exist, "
                                 "please dump meta data 'trip' first.".format(self.get_meta_path('trip', select_set)))

            trip_arr, valid_lens = data_augment_func(trip_arr, valid_lens, edit_ratio, time_index=7, padding=True)

            meta = [trip_arr, valid_lens]

        elif 'transprob' in meta_type:
            """
            Loc transfer probability without adjacency.
            """
            # Gather all possible transfer between road segments.
            known_trip_trans = known_trips.shift(1).join(known_trips, lsuffix='_from', rsuffix='_to')
            known_trip_trans = known_trip_trans[known_trip_trans['trip_from'] == known_trip_trans['trip_to']]
            known_trip_trans = known_trip_trans[known_trip_trans['road_from'] != known_trip_trans['road_to']]
            transfer_counter = Counter((int(row['road_from']), int(row['road_to']))
                                       for index, row in known_trip_trans.iterrows())
            transfer_pairs = pd.DataFrame([[l, r, v] for (l, r), v in transfer_counter.items()], columns=[
                'road_from', 'road_to', 'freq'])

            froms = []
            tos = []
            freqs = []
            for index, group in transfer_pairs.groupby('road_from'):
                for _, sample in group.iterrows():
                    froms.append(int(sample['road_from']))
                    tos.append(int(sample['road_to']))
                    freqs.append(float(sample['freq']) / max(1, sample['freq'].sum()))

            edge_index = np.array([froms, tos])
            trans_prob = np.array(freqs)[:, None]

            meta = [edge_index, trans_prob]

        elif 'trajsim' in meta_type:
            Data_generator = Trajsim_data_generator(self.trips, meta_type, self.meta_dir)
            Data_generator.create_vocal()
            if not os.path.exists(Data_generator.VDpath):
                Data_generator.create_dist()

            src_arr, tgt_arr, mta_arr = Data_generator.create_data(trips, select_trip_id)
            meta = [src_arr, tgt_arr, mta_arr]

        elif 'detourtgt' in meta_type:
            """
            Detour-based similar trajectory search using in START.
            """
            params = meta_type.split('-')
            num_target = int(params[1])
            num_negative = int(params[2])

            self.build_network()

            target_is, tgt_detoured_trips, tgt_detour_lens = [], [], []
            for target_i, (_, group) in tqdm(enumerate(trips.groupby('trip')),
                                             desc="Gathering detour positive meta",
                                             total=len(select_trip_id)):
                # Avoid multi traj points in one segment
                group = group.loc[~group['road'].duplicated(), TRIP_COLS + ['time']]
                group['seq_i'] = list(range(len(group)))

                a, detour_len, detour = traj_detour(group, max_detour_amount=0.2, road_network=self.network,
                                                    road_info=self.road_info)

                if a is None:
                    continue
                else:
                    arr = a[TRIP_COLS].to_numpy()

                    offset = a['time'].apply(lambda x: x.timestamp()).to_numpy()
                    offset = (offset - offset[0]) / (offset[-1] - offset[0]) * 2 - 1
                    arr = np.append(arr, offset.reshape(-1, 1), 1)

                    tgt_detoured_trips.append(arr)
                    tgt_detour_lens.append(len(a))
                    target_is.append(target_i)

                    if len(tgt_detoured_trips) == num_target:
                        break

            target_is = np.array(target_is)
            tgt_detour_lens = np.array(tgt_detour_lens)
            max_tgt_detour_len = tgt_detour_lens.max()
            padded_detour_trips = []
            for group in tgt_detoured_trips:
                padded_detour_trips.append(np.concatenate(
                    [group, np.repeat(group[-1:], max_tgt_detour_len - group.shape[0], axis=0)], 0))
            tgt_detoured_trips = np.stack(padded_detour_trips, 0)
            tgt_detoured_trips = trip_normalizer(tgt_detoured_trips)

            meta = [tgt_detoured_trips, tgt_detour_lens, target_is]

        elif 'detourqry' in meta_type:
            """
            Detour-based similar trajectory search using in START.
            """
            params = meta_type.split('-')
            num_target = int(params[1])
            num_negative = int(params[2])

            try:
                trip_arrs, valid_lens = self.load_meta('trip', select_set)
                _, _, target_is = self.load_meta('detourtgt-{}-{}'.format(num_target, num_negative), select_set)
            except FileNotFoundError:
                raise FileNotFoundError("Dump meta type trip and detourgtgt-{}-{} first."
                                        .format(num_target, num_negative))

            query_trips = trip_arrs[target_is, ...]
            query_valid_lens = valid_lens[target_is, ...]

            meta = [query_trips, query_valid_lens]

        elif 'detourneg' in meta_type:
            """
            Detour-based similar trajectory search using in START.
            """
            params = meta_type.split('-')
            num_target = int(params[1])
            num_negative = int(params[2])

            self.build_network()

            try:
                _, _, target_is = self.load_meta(f'detourtgt-{num_target}-{num_negative}', select_set)
            except FileNotFoundError:
                raise FileNotFoundError(f"Dump meta detourtgt-{num_target}-{num_negative} first.")

            neg_detoured_trips, neg_detour_lens = [], []
            for target_i, (_, group) in tqdm(enumerate(trips.groupby('trip')),
                                             desc="Gathering detour positive meta",
                                             total=len(select_trip_id)):
                if target_i <= target_is[-1]:
                    continue

                # Avoid multi traj points in one segment
                group = group.loc[~group['road'].duplicated(), TRIP_COLS + ['time']]
                group['seq_i'] = list(range(len(group)))

                a, detour_len, detour = traj_detour(group, max_detour_amount=0.2,
                                                    road_network=self.network, road_info=self.road_info)

                if a is None:
                    continue
                else:
                    arr = a[TRIP_COLS].to_numpy()

                    offset = a['time'].apply(lambda x: x.timestamp()).to_numpy()
                    offset = (offset - offset[0]) / (offset[-1] - offset[0]) * 2 - 1
                    arr = np.append(arr, offset.reshape(-1, 1), 1)

                    neg_detoured_trips.append(arr)
                    neg_detour_lens.append(len(a))

                    if len(neg_detoured_trips) == num_negative:
                        break

            neg_detour_lens = np.array(neg_detour_lens)
            max_neg_detour_len = neg_detour_lens.max()
            padded_neg_detour_trips = []
            for group in neg_detoured_trips:
                padded_neg_detour_trips.append(np.concatenate(
                    [group, np.repeat(group[-1:], max_neg_detour_len - group.shape[0], axis=0)], 0))
            neg_detoured_trips = np.stack(padded_neg_detour_trips, 0)
            neg_detoured_trips = trip_normalizer(neg_detoured_trips)

            meta = [neg_detoured_trips, neg_detour_lens]

        elif 'hoptgt' in meta_type:
            """
            Target set for hop-based similar trajectory search,
            target: odd indices
            """
            params = meta_type.split('-')
            num_target = int(params[1])
            num_negative = int(params[2])

            try:
                trips, valid_lens = self.load_meta('trip', select_set)
            except FileNotFoundError:
                raise FileNotFoundError("Dump meta type {} first.".format('trip'))

            # target_valid_lens = np.ceil(valid_lens / 2).astype(int)
            # target_trips = trips[:num_target, ::2, :]

            target_trips, target_valid_lens, target_is = [], [], []
            for i, (trip, valid_len) in tqdm(enumerate(zip(trips, valid_lens)),
                                             desc="Gathering hop-based target samples",
                                             total=len(valid_lens)):
                if valid_len < 2:
                    continue

                target_trip = trip[1::2]
                target_trips.append(target_trip)
                target_valid_lens.append(math.floor(valid_len / 2))
                target_is.append(i)

                if len(target_trips) == num_target:
                    break

            target_trips = np.stack(target_trips, 0)
            target_valid_lens = np.array(target_valid_lens)
            target_is = np.array(target_is).astype(int)

            meta = [target_trips, target_valid_lens, target_is]

        elif 'hopqry' in meta_type:
            """
            Query set for hop-based similar trajectory search,
            query: even indices
            """
            params = meta_type.split('-')
            num_target = int(params[1])
            num_negative = int(params[2])

            try:
                _, _, target_is = self.load_meta('hoptgt-{}-{}'.format(num_target, num_negative), select_set)
                trips, valid_lens = self.load_meta('trip', select_set)
            except FileNotFoundError:
                raise FileNotFoundError("Dump meta type {} first.".format('trip'))

            query_valid_lens = np.ceil(valid_lens / 2).astype(int)
            query_trips = trips[target_is, ::2, :]

            meta = [query_trips, query_valid_lens]

        elif 'hopneg' in meta_type:
            params = meta_type.split('-')
            num_target = int(params[1])
            num_negative = int(params[2])

            try:
                _, _, target_is = self.load_meta('hoptgt-{}-{}'.format(num_target, num_negative), select_set)
                trips, valid_lens = self.load_meta('trip', select_set)
            except FileNotFoundError:
                raise FileNotFoundError("Dump meta type trip and hoptgt-{}-{} first.".format(num_target, num_negative))

            max_index = target_is[-1]
            neg_trips = trips[max_index: max_index + num_negative]
            neg_valid_lens = valid_lens[max_index: max_index + num_negative]

            meta = [neg_trips, neg_valid_lens]

        elif 'slice' in meta_type:
            """
            Slice data augmentation,
            Wang H, Feng J, Sun L, et al. 
            Abnormal trajectory detection based on geospatial consistent modeling. IEEE Access, 2020
            """
            from utils import SDMSampleGenerator

            params = meta_type.split('-')
            window_size = int(params[1])

            arrs = []
            for _, group in tqdm(trips.groupby('trip'), desc="Gathering trip slices", total=len(select_trip_id)):
                arr = group['road'].to_numpy().astype(int)
                mask = [i for i in range(1, len(arr)) if arr[i] == arr[i - 1]]
                unmask = [True if e not in mask else False for e in np.arange(len(arr))]
                arr = arr[unmask]

                valid_len = arr.shape[0]
                end_index = valid_len - window_size + 1

                index = 0
                while index < end_index:
                    arrs.append(arr[index: index + window_size])
                    index += 1
            arrs = np.stack(arrs, 0)

            generator = SDMSampleGenerator(arrs, min_count=5)

            sources = []
            destinations = []
            midways = []
            negs = []
            for arr in arrs:
                source = arr[0]
                destination = arr[-1]
                midway = arr[1:-1]
                for m in midway:
                    sources.append(source)
                    destinations.append(destination)
                    midways.append(m)
                    negs.append(generator.getNegatives(m, 5))
            sources, destinations, midways, negs = [np.array(e) for e in (sources, destinations, midways, negs)]

            meta = [sources, destinations, midways, negs]

        elif 'trajimage' in meta_type:
            params = meta_type.split('-')
            threshold = int(params[1])
            dest_pre_length = int(params[2]) if len(params) == 3 else None

            images = []
            for _, group in tqdm(trips.groupby('trip'), desc="Gathering traj images", total=len(select_trip_id)):
                if dest_pre_length is not None:
                    group = group.loc[:len(group) - dest_pre_length, :]

                image = np.zeros([self.num_h, self.num_w])
                magntitudes = Counter(group['road'])
                for road, count in magntitudes.items():
                    if count < threshold:
                        continue

                    h = road // self.num_w
                    w = road % self.num_w
                    image[h, w] = 1
                images.append(image)

            images = np.stack(images, axis=0)[:, None, ...]  # (N, C, H, W)

            meta = [images, ]

        elif 'coccur' in meta_type:
            """
            Co-occurrence in time window
            """
            # Parameters are given in the meta_type. Use format "coccur-{window_size}".
            params = meta_type.split('-')
            window_size = int(params[1])

            pos_samples = []
            neg_samples = []
            for _, group in tqdm(trips.groupby('trip'), desc='Gathering co-occurence pairs', total=len(select_trip_id)):
                # Avoid multi traj points in one segment
                group = group.loc[~group['road'].duplicated(), TRIP_COLS + ['time']]
                group['seq_i'] = list(range(len(group)))

                for t, row in group.iterrows():
                    window = group.loc[((group['time'] - row['time']).apply(lambda x: x.total_seconds()) <= window_size) &
                                       ((group['time'] - row['time']).apply(lambda x: x.total_seconds()) >= 0), 'road'].to_list()
                    if len(window) < 2:
                        continue

                    pos_pairs = np.array(list(combinations(window, 2)))
                    pos_samples.append(pos_pairs)

                    neg_pairs = deepcopy(pos_pairs)
                    neg_pairs[range(len(neg_pairs)), np.random.randint(2, size=len(neg_pairs))] = \
                        np.random.choice(list(set(self.road_info['road'].to_list()).difference(set(window))),
                                         len(neg_pairs))
                    neg_samples.append(neg_pairs)

            pos_samples = np.concatenate(pos_samples, axis=0)
            neg_samples = np.concatenate(neg_samples, axis=0)

            pos_df = pd.DataFrame(pos_samples, columns=['r1', 'r2'])
            pos_df['coccur'] = 1
            pos_df = pd.merge(pos_df, self.road_info[['road', 'level']], left_on='r1', right_on='road', how='left')
            pos_df = pd.merge(pos_df, self.road_info[['road', 'level']], left_on='r2', right_on='road', how='left', suffixes=('_1', '_2'))
            same_type = [1 if t1 == t2 else 0 for t1, t2 in zip(pos_df['level_1'], pos_df['level_2'])]
            pos_df['type'] = same_type

            neg_df = pd.DataFrame(neg_samples, columns=['r1', 'r2'])
            neg_df['coccur'] = 0
            neg_df = pd.merge(neg_df, self.road_info[['road', 'level']], left_on='r1', right_on='road', how='left')
            neg_df = pd.merge(neg_df, self.road_info[['road', 'level']], left_on='r2', right_on='road', how='left',
                              suffixes=('_1', '_2'))
            same_type = [1 if t1 == t2 else 0 for t1, t2 in zip(neg_df['level_1'], neg_df['level_2'])]
            neg_df['type'] = same_type

            meta = [pos_df.loc[:, ['r1', 'r2', 'coccur', 'type']].values,
                    neg_df.loc[:, ['r1', 'r2', 'coccur', 'type']].values]

        elif 'road2vec' in meta_type:
            """
            Road2Vec module, return the road segment embeddings. 
            """
            params = meta_type.split('-')
            window_size = int(params[1])
            embed_dim = int(params[2])

            try:
                pos_samples, neg_samples = self.load_meta('coccur-{}'.format(window_size), select_set)
            except FileNotFoundError:
                raise FileNotFoundError("Dump coccur-{} first.".format(window_size))

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            alpha_ = 0.9

            road2vec = Road2Vec(self.num_road, embed_dim).to(device)
            samples = np.concatenate([pos_samples, neg_samples], axis=0).astype(float)
            coccur_criterion = torch.nn.BCELoss()
            type_criterion = torch.nn.BCELoss()
            optimizer = torch.optim.Adam(road2vec.parameters(), lr=1e-3)

            mean_loss = float('inf')
            loss_log = []
            description = 'Road2Vec training, loss: %.4f'
            with trange(100, desc=description % mean_loss, total=100) as pbar:
                for epoch in pbar:
                    loss_log_epoch = []
                    np.random.shuffle(samples)

                    for batch in next_batch(samples, 512):
                        batch = torch.from_numpy(batch).long().to(device)
                        x = batch[:, :2]
                        logits = road2vec(x)

                        coccur_label = batch[:, 2].float()
                        type_label = batch[:, 3].float()
                        coccur_loss = coccur_criterion(logits, coccur_label)
                        type_loss = type_criterion(logits, type_label)
                        loss = alpha_ * coccur_loss + (1 - alpha_) * type_loss

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        loss_log_epoch.append(loss.item())
                    mean_loss = float(np.mean(loss_log_epoch))
                    pbar.set_description(description % mean_loss)
                    loss_log.append(mean_loss)

            meta = [road2vec.embedding.weight.cpu().detach().numpy()]

        elif 't2vec' in meta_type:
            Data_generator = t2vec_data_generator(self.trips, meta_type, self.meta_dir)
            num_out_region = Data_generator.make_vocab()
            # Data_generator.create_dist()
            if not os.path.exists(Data_generator.VDpath):
                Data_generator.saveKNearestVocabs()

            # Create dataset (Note: Padding operation is not performed here)
            src_arr, tgt_arr, mta_arr = Data_generator.processe_data(trips, select_trip_id)
            meta = [src_arr, tgt_arr, mta_arr]

        elif 'timefea' in meta_type:
            """ 
            Time features including week, day, hour, minute and seconds.
            """
            try:
                trip_arrs, valid_lens = self.load_meta('trip', select_set)
            except FileNotFoundError:
                raise FileNotFoundError("Dump meta trip first.")

            second_seq = trip_arrs[..., 7]
            N, L = second_seq.shape
            timestamps = [pd.to_datetime(e, unit='s') for e in second_seq.reshape(-1)]
            week_seq = np.array([e.weekofyear for e in timestamps])
            day_seq = np.array([e.dayofyear for e in timestamps])
            hour_seq = np.array([e.hour for e in timestamps])
            minute_seq = np.array([e.minute for e in timestamps])
            second_seq = np.array([e.second for e in timestamps])

            out_seq = np.stack([week_seq, day_seq, hour_seq, minute_seq, second_seq], -1).reshape([N, L, 5])

            meta = [out_seq]

        elif 'quadkey' in meta_type:
            """
            Quadkey of road from hierarchical map gridding method.
            """
            params = meta_type.split('-')
            level = int(params[1])

            try:
                trip_arrs, valid_lens = self.load_meta('trip', select_set)
            except FileNotFoundError:
                raise FileNotFoundError("Dump meta trip first.")

            N, L, E = trip_arrs.shape
            min_lat, max_lat = trip_arrs[:, :, 4].min(), trip_arrs[:, :, 4].max()
            min_lon, max_lon = trip_arrs[:, :, 3].min(), trip_arrs[:, :, 3].max()
            quadkeys = [latlon2quadkey(lat, lon, level, min_lat, max_lat, min_lon, max_lon)
                        for lat, lon in zip(trip_arrs[..., 4].reshape(-1), trip_arrs[..., 3].reshape(-1))]
            quadkeys = np.array([list(quadkey) for quadkey in quadkeys]).astype(int).reshape([N, L, -1])

            meta = [quadkeys, valid_lens]

        elif 'robustDAA' in meta_type:
            params = meta_type.split('-')
            width, height = int(params[1]), int(params[2]) # the width and height of the whole grid region
            if len(params) > 3:
                noise = float(params[3]) # the variance of gaussian noise added to trajectory
            else:
                noise = None
            
            def trip2grid(trip: pd.DataFrame):
                # records number of times that the moving object stays in the grid area.
                grid_map = np.zeros((height,width),dtype=np.int32)

                min_lng, min_lat = trip["lng"].min(), trip["lat"].min()
                lng_span = trip["lng"].max()-trip["lng"].min()
                lat_span = trip["lat"].max()-trip["lat"].min()
                for i in trip.index:
                    xoffset = math.floor(width * (trip.loc[i]["lng"]-min_lng)/lng_span)
                    yoffset = math.floor(height * (trip.loc[i]["lat"]-min_lat)/lat_span)
                    # Ensure not out of bounds
                    xoffset = min(xoffset,width-1)
                    yoffset = min(yoffset,height-1)
                    
                    grid_map[yoffset, xoffset] += 1 # Increment count for corresponding grid area

                return grid_map


            src_arrs, tgt_arrs = [], []
            for _, group in tqdm(trips.groupby('trip'), desc='Gathering trips', total=len(select_trip_id)):
                arr = group[['lng', 'lat']]
                # Convert longitude and latitude to meters
                arr[['lng', 'lat']] = arr.apply(lambda row: pd.Series(TransferFunction.lonlat2meters(row['lng'],row['lat'])), axis = 1)
                # Add Gaussian noise to trajectory
                if noise is not None:
                    noise_arr = arr.applymap(lambda x: x + random.gauss(mu=0,sigma=noise))
                else:
                    noise_arr = arr.deepcopy()
                
                grid_features = np.diagonal(trip2grid(arr))
                noise_grid_features = np.diagonal(trip2grid(noise_arr))

                # grid_features is of equal length, no padding needed
                src_arrs.append(noise_grid_features)
                tgt_arrs.append(grid_features)

            src_arrs, tgt_arrs = np.stack(src_arrs,0),np.stack(tgt_arrs,0)
            meta = [src_arrs, tgt_arrs]

        elif 'speedacc' in meta_type:
            """
            Src lng and lat, with speed and acceleration.
            """

            try:
                trip_arrs, valid_lens = self.load_meta('trip', select_set)
            except FileNotFoundError:
                raise FileNotFoundError("Dump meta type trip first.")

            valid_idx = valid_lens > 2
            valid_lens = valid_lens[valid_idx]
            trip_arrs = trip_arrs[valid_idx]

            trip_denormalizer = Denormalizer(self.stat, feat_cols=[0, 2, 3, 4], norm_type='minmax')
            denorm_trips = trip_denormalizer([3, 4], trip_arrs)

            B, L, _ = trip_arrs.shape

            seconds = trip_arrs[:, :, 7]

            speed_arrs, acc_arrs = [], []
            for i in trange(B, desc="Gathering speed and acceleration"):
                dists = [geo_distance(lng1, lat1, lng2, lat2) for lng1, lat1, lng2, lat2 in
                          zip(denorm_trips[i, :-1, 3], denorm_trips[i, :-1, 4],
                              denorm_trips[i, 1:, 4], denorm_trips[i, 1:, 4])]
                time_diff = [seconds[i, j] - seconds[i, j - 1] for j in range(1, L)]
                speeds = [d / t if abs(t) > 1e-2 else 0 for d, t in zip(dists, time_diff)]
                speeds.append(speeds[-1])

                speed_diff = [speeds[j] - speeds[j - 1] for j in range(1, L)]
                accs = [s / t if abs(t) > 1e-2 else 0 for s, t in zip(speed_diff, time_diff)]
                accs.append(accs[-1])

                speed_arrs.append(speeds)
                acc_arrs.append(accs)
            speed_arrs = np.array(speed_arrs)[:, :, None]
            acc_arrs = np.array(acc_arrs)[:, :, None]

            meta = [np.concatenate([trip_arrs[..., [3, 4]], speed_arrs, acc_arrs], axis=2).astype(np.float32),
                    valid_lens - 2]

        elif 'ksegsimidx' in meta_type:
            """ Indices of K-segment based similar trajectory search. """
            params = meta_type.split('-')
            num_target = int(params[1])
            num_negative = int(params[2])

            try:
                trip_arrs, valid_lens = self.load_meta('trip', select_set)
            except FileNotFoundError:
                raise FileNotFoundError("Dump meta type trip first.")

            anchor = np.array([-0.5, -0.5])[None, :]
            min_distances_to_anchor = np.sqrt(((trip_arrs[:, :, [3, 4]] - anchor) ** 2).sum(-1)).min(-1)
            neg_location_index = min_distances_to_anchor > 1
            min_distances_to_anchor = np.sqrt(((trip_arrs[:, :, [3, 4]] - anchor) ** 2).sum(-1)).max(-1)
            qry_location_index = min_distances_to_anchor < 0.5

            qry_index = qry_location_index
            neg_index = neg_location_index

            qry_index = np.arange(len(trip_arrs))[qry_index]
            neg_index = np.arange(len(trip_arrs))[neg_index]
            query_trips = trip_arrs[qry_index]
            query_valid_lens = valid_lens[qry_index]

            qry_seg_trips = resample_to_k_segments(query_trips[..., [3, 4]], query_valid_lens, MIN_TRIP_LEN)
            ball_tree = BallTree(qry_seg_trips)

            dist, index = ball_tree.query(qry_seg_trips, 2)
            tgt_index_in_qry = index[:, 1]
            tgt_index = qry_index[tgt_index_in_qry]

            qry_index = qry_index[:num_target]
            tgt_index = tgt_index[:num_target]
            neg_index = neg_index[:num_negative]
            print("Query trip num: {}\n Target trip num: {}\n Negative trip num: {}"
                  .format(len(qry_index), len(tgt_index), len(neg_index)))

            meta = [qry_index, tgt_index, neg_index]

        else:
            raise NotImplementedError('No meta type', meta_type)

        create_if_noexists(self.meta_dir)
        meta_path = self.get_meta_path(meta_type, select_set)
        np.savez(meta_path, *meta)
        print('Saved meta to', meta_path)


def gen_time_mat(time_array: np.array):
    """
    Args:
        time_array: (num_trips, max_len)
    """
    num_trips, max_len = time_array.shape

    subtrahend = repeat(time_array, "N L -> N L L2", L2=max_len)
    minuend = repeat(time_array, "N L -> N L2 L", L2=max_len)
    off = abs(minuend - subtrahend)

    return off


class Normalizer(nn.Module):
    def __init__(self, stat, feat_cols, feat_names=None, norm_type='zscore'):
        super().__init__()

        self.stat = stat
        self.feat_cols = feat_cols
        self.feat_names = feat_names if feat_names is not None \
            else [TRIP_COLS[feat_col] for feat_col in feat_cols]
        self.norm_type = norm_type

    def _norm_col(self, x_col, col_name):
        if self.norm_type == 'zscore':
            x_col = (x_col - self.stat.loc['mean', col_name]) / self.stat.loc['std', col_name]
        elif self.norm_type == 'minmax':
            x_col = (x_col - self.stat.loc['min', col_name]) / \
                    (self.stat.loc['max', col_name] - self.stat.loc['min', col_name])
            x_col = x_col * 2 - 1
        else:
            raise NotImplementedError(self.norm_type)
        return x_col

    def forward(self, arr):
        """ Normalize the input array. """
        if isinstance(arr, torch.Tensor):
            x = torch.clone(arr)
        else:
            x = np.copy(arr)
        for col, name in zip(self.feat_cols, self.feat_names):
            x[..., col] = self._norm_col(x[..., col], name)
        return x


class Denormalizer(nn.Module):
    def __init__(self, stat, feat_cols, feat_names=None, norm_type='zscore'):
        super().__init__()

        self.stat = stat
        self.feat_names = feat_names
        self.feat_cols = feat_cols
        self.feat_names = feat_names if feat_names is not None \
            else [TRIP_COLS[feat_col] for feat_col in feat_cols]
        self.norm_type = norm_type

    def _denorm_col(self, x_col, col_name):
        if self.norm_type == 'zscore':
            x_col = x_col * self.stat.loc['std', col_name] + self.stat.loc['mean', col_name]
        elif self.norm_type == 'minmax':
            x_col = (x_col + 1) / 2
            x_col = x_col * (self.stat.loc['max', col_name] - self.stat.loc['min', col_name]) + \
                    self.stat.loc['min', col_name]
        else:
            raise NotImplementedError(self.norm_type)
        return x_col

    def forward(self, select_cols, arr):
        """ Denormalize the input batch. """
        if isinstance(arr, torch.Tensor):
            x = torch.clone(arr)
        else:
            x = np.copy(arr)
        for col, name in zip(self.feat_cols, self.feat_names):
            if col in select_cols:
                x[..., col] = self._denorm_col(x[..., col], name)
        return x


def gen__mat(time_array: np.array):
    """
    Args:
        time_array: (num_trips, max_len)
    """
    num_trips, max_len = time_array.shape

    subtrahend = repeat(time_array, "N L -> N L L2", L2=max_len)
    minuend = repeat(time_array, "N L -> N L2 L", L2=max_len)
    off = abs(minuend - subtrahend)

    return off


def trim_traj(trips: np.array, valid_len, edit_ratio=0.1, trim_od_only=True, **kwargs):
    """
    Delete several traj points from trajs
    """
    padding = kwargs.get('padding', False)
    num_trajs, max_len, num_feats = trips.shape

    if not trim_od_only:
        trim_idx = np.array([np.random.choice(range(i), math.ceil(edit_ratio * i)) if i > math.ceil(edit_ratio * i)
                             else np.random.choice(range(i), i - 1, replace=False) for i in valid_len])

    res_trajs = []
    trim_length = 0
    random_num = np.random.rand()
    for i, length in enumerate(valid_len):
        select_idx = np.ones_like(trips[0])
        if math.ceil(edit_ratio * length) >= length:
            trim_length = length - 1
            if not trim_od_only:
                select_idx[trim_idx[i], :] = 0
            else:
                if random_num > 0.5:
                    select_idx[0: length - 1, :] = 0
                else:
                    select_idx[1: length, :] = 0
        else:
            trim_length = math.ceil(edit_ratio * length)
            if not trim_od_only:
                select_idx[trim_idx[i], :] = 0
            else:
                if random_num > 0.5:
                    select_idx[0: trim_length, :] = 0
                else:
                    select_idx[length - trim_length: length, :] = 0
        select_idx = select_idx.astype(bool)
        res_traj = trips[i][select_idx].reshape(max_len - trim_length, num_feats)
        if padding:
            res_traj = np.concatenate([res_traj, np.repeat(res_traj[-1:], trim_length, axis=0)], axis=0)
        res_trajs.append(res_traj)
        valid_len[i] -= trim_length

    res_trajs = np.stack(res_trajs, axis=0)
    return res_trajs, valid_len


def temporal_shift(trips: np.array, valid_len, edit_ratio=0.15, time_index=7, **kwargs):
    num_trajs, max_len, num_feats = trips.shape

    # select traj points to be shifted, not including the first and last ones
    perturb_indices = [
        np.random.choice(range(1, i - 1), math.ceil(edit_ratio * i), replace=False) if i > math.ceil(edit_ratio * i) + 2
        else np.random.choice(range(1, i), i - 1, replace=False) for i in valid_len]

    time_seq = trips[:, :, int(time_index)]
    for i, indices in enumerate(perturb_indices):
        for idx in indices:
            time_seq[i, idx] = time_seq[i, idx] - \
                               (time_seq[i, idx] - time_seq[i, idx - 1]) * (
                                           np.random.rand() * 0.15 + 0.15)  # shift by a random amount, Uniform(t_{i-1}, t_{i+1})
            time_seq[i, idx] = round(time_seq[i, idx])  # maybe it can be float?
    trips[:, :, int(time_index)] = time_seq  # tod

    return trips, valid_len


class Road2Vec(nn.Module):
    def __init__(self, num_roads, embed_dim):
        super(Road2Vec, self).__init__()

        self.embedding = nn.Embedding(num_roads, embed_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        uxy = self.embedding(x)
        ux, uy = uxy[:, 0], uxy[:, 1]
        return self.sigmoid(torch.sum(torch.multiply(ux, uy), dim=1))


def traj_detour(trip, max_detour_amount,
                road_network: nx.Graph, road_info):
    trip_len = len(trip)
    if trip_len < 2:
        return None, None, None

    max_detour_amount = np.clip(max_detour_amount, 0, 1)
    detour_amount = np.clip(max_detour_amount, 0, 1)
    detour_amount = math.ceil(trip_len * detour_amount)

    detour_len = max(2, detour_amount)
    detour_start = random.randint(0, trip_len - detour_len)

    # An odd way to determine the origin and destination of a road in the traj.
    o, d = int(trip.loc[trip['seq_i'] == detour_start, 'road']), \
           int(trip.loc[trip['seq_i'] == detour_start + detour_len - 1, 'road'])
    o_back, d_forward = int(trip.loc[trip['seq_i'] == detour_start + 1, 'road']), \
                        int(trip.loc[trip['seq_i'] == detour_start + detour_len - 2, 'road'])

    o = list(set(road_info.loc[road_info['road'] == o, ['o', 'd']].values[0].tolist()) -
             set(road_info.loc[road_info['road'] == o_back, ['o', 'd']].values[0].tolist()))
    if len(o) == 0:
        o = int(road_info.loc[road_info['road'] == o, 'o'].item())
    else:
        o = o[0]

    d = list(set(road_info.loc[road_info['road'] == d, ['o', 'd']].values[0].tolist()) -
             set(road_info.loc[road_info['road'] == d_forward, ['o', 'd']].values[0].tolist()))
    if len(d) == 0:
        d = int(road_info.loc[road_info['road'] == d, 'd'].item())
    else:
        d = d[0]

    o_time, d_time = trip.loc[trip['seq_i'] == detour_start, 'time'].item().timestamp(), \
                     trip.loc[trip['seq_i'] == detour_start + detour_len - 1, 'time'].item().timestamp()
    src_detour_part = trip.loc[(trip['seq_i'] >= detour_start) &
                               (trip['seq_i'] <= detour_start + detour_len - 1), :]
    src_head_part = trip.loc[trip['seq_i'] < detour_start, :]
    src_tail_part = trip.loc[trip['seq_i'] > detour_start + detour_len - 1, :]

    src_detour_part = pd.merge(src_detour_part, road_info[['road', 'length']],
                               on='road', how='left')
    src_detour_length = src_detour_part['length'].sum() / 1000

    src_part = pd.merge(trip, road_info[['road', 'length']], on='road', how='left')
    src_length = src_part['length'].sum() / 1000

    try:
        paths = k_shortest_paths(road_network, o, d, k=10, weight='length')
        # lens, paths = k_shortest_paths(road_network, o, d, k=10, weight='length')  # custom Yen KSP
        res_detour = None
        min_detour_rate = 1
        for i, path in enumerate(paths):
            de_len = len(path)
            if de_len < 2:
                return None, None, None
            os = [min(int(path[i]), int(path[i + 1])) for i in range(de_len - 1)]
            ds = [max(int(path[i]), int(path[i + 1])) for i in range(de_len - 1)]
            detour_times = np.linspace(o_time, d_time, de_len-1).astype(int)
            detour = pd.DataFrame([detour_times.tolist(), os, ds]).T
            detour.columns = ['time', 'o', 'd']
            detour = pd.merge(detour, road_info[['road', 'o', 'd', 'road_lng', 'road_lat', 'length']],
                              on=['o', 'd'], how='left')

            detour_length = detour['length'].sum() / 1000

            detour_rate = abs(detour_length - src_detour_length) / src_length
            if detour_rate > max_detour_amount:
                if detour_rate < min_detour_rate:
                    min_detour_rate = detour_rate
                    res_detour = detour

        if res_detour is None:
            return None, None, None

    except nx.NetworkXNoPath:
        return None, None, None

    res_detour['time'] = pd.to_datetime(res_detour['time'], unit='s')
    res_detour['road_prop'] = 0.5
    res_detour['seq_i'] = list(range(len(res_detour)))
    res_detour['seconds'] = res_detour['time'].apply(lambda x: x.timestamp())
    res_detour['tod'] = res_detour['seconds'] % (24 * 60 * 60) / (24 * 60 * 60)
    res_detour['weekday'] = res_detour['time'].dt.weekday
    res_detour.rename(columns={'road_lng': 'lng', 'road_lat': 'lat'}, inplace=True)

    a = pd.concat([src_head_part, res_detour, src_tail_part])
    a['seq_i'] = list(range(len(a)))
    # a['time'] = pd.to_datetime(a['time'], unit='s')
    res_detour = res_detour.loc[:, TRIP_COLS]

    return a, detour_len, res_detour


class Trajsim_data_generator():
    def __init__(self, trips, meta_type, meta_dir):
        # trips: All trips in the dataset!!!!
        self.trips = trips
        params = meta_type.split('-')
        scale, time_size, hot_freq = float(params[1]), int(params[2]), int(params[3])
        self.args = self.get_trajsim_args(trips, scale, time_size, hot_freq)
        self.transferFunction = TransferFunction(self.args)
        self.VDpath = os.path.join(meta_dir, meta_type + "_dist.h5")
        self.dropping_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.distorting_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        # Map mapping from map_id to hot words
        self.mapId2nnword = {}
        self.trips_starttime = self.trips["time"].apply(lambda x: x.timestamp()).min()

    @staticmethod
    def get_trajsim_args(trips, scale, time_size, hot_freq):
        args = {}
        args["hot_freq"] = hot_freq  # Minimum frequency for hot words, 30
        args["space_nn_topK"] = 20
        args["PAD"], args["BOS"], args["EOS"], args["UNK"] = 0, 1, 2, 3
        args["scale"], args["time_size"] = scale, time_size  # Latitude/longitude division unit (e.g., 0.001°), number of time cells
        args["lons"] = [trips["lng"].min(), trips["lng"].max()]  # Range of longitude
        args["lats"] = [trips["lat"].min(), trips["lat"].max()]  # Range of latitude
        # Maximum spatial x-coordinate number, maximum spatial y-coordinate number
        args["maxx"], args["maxy"] = (args["lons"][1] - args["lons"][0]) // args["scale"], (
                args["lats"][1] - args["lats"][0]) // args["scale"]
        # Number of horizontal blocks in space, number of vertical blocks in space
        args["numx"], args["numy"] = args["maxx"], args["maxy"]
        # Number of spatial cells
        args["space_cell_size"] = args["maxx"] * args["maxy"]
        # Length of each time period
        args["time_span"] = 86400 // args["time_size"]
        # Vocabulary word encoding starts from 4, 0,1,2,3 have special purposes
        args["start"] = args["UNK"] + 1
        # Number of spatial neighbors when encoding a spatiotemporal grid, used for generating V, D
        args["space_nn_nums"] = 20
        # Number of temporal neighbors when encoding a spatiotemporal grid, also the final number of spatiotemporal neighbors, used for generating V, D
        args["time_nn_nums"] = 10
        # Number of spatiotemporal grid cells (x,y,t three dimensions)
        args["map_cell_size"] = args["maxx"] * args["maxy"] * args["time_size"]

        return DotDict(args)

    def create_data(self, trips, select_trip_id):
        src_arr, tgt_arr, mta_arr = [], [], []
        # select_set: index of set to dump. 0 - training set, 1 - validation set, and 2 - testing set
        for _, group in tqdm(trips.groupby('trip'), desc='Gathering trips', total=len(select_trip_id)):
            trip = group[['lng', 'lat']].to_numpy()
            ts = group['time'].apply(lambda x: x.timestamp() - self.trips_starttime).to_numpy()

            mta = self.transferFunction.tripmeta(trip, ts)  # tuple
            # Trajectory spatiotemporal encoding sequence, all hot words, effectively reducing number of tokens
            exact_trj = self.transferFunction.trip2words(trip, ts,
                                                         self.hotcell2word, self.tree, self.hot_ts)  # list
            # Based on original trajectory, only update positions with downsampling and offset, improving efficiency
            noise_trips, noise_ts, noise_trjs = self.add_noise(exact_trj, trip, ts)
            for src in noise_trjs:
                src_arr.append(src)  # ndarray
                tgt = [self.args.BOS] + exact_trj + [self.args.EOS]
                tgt_arr.append(np.array(tgt))
                mta_arr.append(np.array(mta))

        return src_arr, tgt_arr, mta_arr

    def downsamplingDistort(self, trip):
        """
        Downsampling + adding noise, only recalculate hot word encoding for points with downsampling and offset
        """
        noisetrips = []
        for dropping_rate in self.dropping_rates:
            noisetrip1 = self.downsampling(trip, dropping_rate)
            for distorting_rate in self.distorting_rates:
                noisetrip2 = self.distort(noisetrip1, distorting_rate)
                noisetrips.append(noisetrip2)
        return noisetrips


class t2vec_data_generator():
    def __init__(self, trips, meta_type, meta_dir):
        self.trips = trips
        params = meta_type.split('-')
        # t2vec code settings: 100,100,10,100,40000
        xstep, ystep, K, hot_freq, maxvocab_size = int(params[1]), int(params[2]), \
                                                   int(params[3]), int(params[4]), int(params[5])
        self.region = self.get_region(trips, xstep, ystep, K, hot_freq, maxvocab_size)
        self.transferFunction = TransferFunction(self.region)

        self.VDpath = os.path.join(meta_dir, meta_type+"_dist.h5")
        self.cellcount = {}
        self.hotcell = []
        self.hotcell_kdtree = None
        self.hotcell2vocab = {}
        self.vocab2hotcell = {}
        self.vocab_start = self.region.vocab_start
        self.vocab_size = self.region.vocab_start

        self.dropping_rates = [0, 0.2, 0.4, 0.5, 0.6]
        self.distorting_rates = [0, 0.2, 0.4, 0.6]

    @staticmethod
    def get_region(trips, xstep, ystep, K, hot_freq, maxvocab_size):
        region = {}

        region["lons"] = [trips["lng"].min(), trips["lng"].max()]  # range of longitude
        region["lats"] = [trips["lat"].min(), trips["lat"].max()]  # range of latitude
        # Longitude and latitude extremes corresponding to (x,y) (unit: meters)
        region["minxy"] = list(TransferFunction.lonlat2meters(region["lons"][0], region["lats"][0])) # tuple to list
        region["maxxy"] = list(TransferFunction.lonlat2meters(region["lons"][1], region["lats"][1]))

        region["xstep"], region["ystep"] = xstep, ystep  # Grid length/width
        # Number of x/y direction grids in space
        region["numx"] = int(math.ceil((region["maxxy"][0] - region["minxy"][0]) / region["xstep"]))  # Number of x direction grids
        region["numy"] = int(math.ceil((region["maxxy"][1] - region["minxy"][1]) / region["ystep"]))  # Number of y direction grids

        region["hot_freq"] = hot_freq  # Minimum occupancy frequency for hot words, 30
        region["maxvocab_size"] = maxvocab_size  # Maximum size of hot word table

        region["K_nearest"] = K

        region["PAD"], region["BOS"], region["EOS"], region["UNK"] = 0, 1, 2, 3
        # Vocabulary word encoding starts from 4, 0,1,2,3 have special purposes
        region["vocab_start"] = region["UNK"] + 1

        return DotDict(region)


    def make_vocab(self):
        '''
        Build vocabulary, obtain hot words and mappings between hot words and vocabulary
        Count and dictionary mapping conversion, convert hot words to coordinates
        '''
        print("Creating vocabulary, obtaining hot word mappings")
        self.cellcount = defaultdict(int)  # Build dictionary to count cell_id
        num_out_region = 0

        # Count encoding values
        for _, group in tqdm(self.trips.groupby('trip'), desc='Gathering trips'):
            trip = group[['lng', 'lat']].to_numpy()
            for lon, lat in trip:
                if not self.transferFunction.if_inRegion(lon, lat):
                    num_out_region += 1
                else:
                    cell = self.transferFunction.lonlat2cellId(lon, lat)
                    self.cellcount[cell] += 1

        max_num_hotcells = min(self.region.maxvocab_size, len(self.cellcount))
        topcellcount = sorted(self.cellcount.items(), key=lambda x: x[1], reverse=True)[:max_num_hotcells]

        self.hotcell = [cell for cell, count in topcellcount if count >= self.region.hot_freq]
        self.hotcell2vocab = {cell: i + self.vocab_start for i, cell in enumerate(self.hotcell)}
        self.vocab2hotcell = {count: cell for cell, count in self.hotcell2vocab.items()}
        self.vocab_size = self.vocab_start + len(self.hotcell)
        print('Vocabulary created, total number of vocabulary (including 4 special codes): {}'.format(self.vocab_size))

        # Build Kdtree based on coordinates of hot words
        coord = [self.transferFunction.cellId2coord(cellId) for cellId in self.hotcell]
        self.hotcell_kdtree = spatial.cKDTree(coord)

        return num_out_region


    def saveKNearestVocabs(self):
        """
        k-nearest vocabs and corresponding distances for each vocab.
        This is used in training for KLDiv loss.
        """
        print("Using constructed Kdtree to calculate distances between all hot words, building spatial neighbors V,D for each hot word")
        V = np.zeros((self.vocab_size, self.region.K_nearest), dtype=np.int32)  # k-nearest vocabs (vocab_size, k)
        D = np.zeros((self.vocab_size, self.region.K_nearest), dtype=np.float64)  # k-nearest distances (vocab_size, k)

        for vocab in range(self.vocab_start):  # Special words distance set to 0
            V[vocab, :] = vocab
            D[vocab, :] = 0.0

        # Get space_nn_nums closest neighbors and their distances for hot words in space
        for vocab in range(self.vocab_start, self.vocab_size):
            cellId = self.vocab2hotcell[vocab]
            kcells, dists = self.transferFunction.knearestHotcells(cellId, self.region.K_nearest, self.hotcell_kdtree, self.hotcell)
            kvocabs = [self.hotcell2vocab[x] for x in kcells]
            V[vocab, :] = kvocabs
            D[vocab, :] = dists

        cellsize = int(self.region.xstep)
        # Store spatial K-nearest neighbors and space distances, write to h5 files
        with h5py.File(self.VDpath, "w") as f:
            f["V"] = V
            f["D"] = D
        print("Saved cell distance into", self.VDpath)


    def processe_data(self, trips, select_trip_id):
        src_arr, tgt_arr, mta_arr = [], [], []
        # select_set: index of set to dump. 0 - training set, 1 - validation set, and 2 - testing set.
        for _, group in tqdm(trips.groupby('trip'), desc='Gathering trips', total=len(select_trip_id)):
            trip = group[['lng', 'lat']].to_numpy()
            meta = self.transferFunction.get_tripmeta(trip)
            # Trajectory spatiotemporal encoding sequence, all hot words, effectively reducing number of tokens
            exact_traj = self.transferFunction.trip2seq(trip, self.hotcell2vocab,
                                                        self.hotcell_kdtree, self.hotcell)
            # Based on original trajectory, only update positions with downsampling and offset, improving efficiency
            noise_trips = self.downsamplingDistort(trip)
            for noise_trip in noise_trips:
                noise_seq = self.transferFunction.trip2seq(noise_trip, self.hotcell2vocab,
                                                        self.hotcell_kdtree, self.hotcell)
                src_arr.append(np.array(noise_seq))
                # Same trajectory, different perturbation and deletion conditions, their trg and meta are the same
                tgt = [self.region.BOS] + exact_traj + [self.region.EOS]
                tgt_arr.append(np.array(tgt))
                mta_arr.append(np.array(meta))

        return src_arr, tgt_arr, mta_arr

    def distort(self, trip, rate, radius=50.0):
        noisetrip = trip.copy()
        for i in range(trip.shape[0]):
            if np.random.rand() <= rate:
                x, y = TransferFunction.lonlat2meters(*trip[i, :])
                xnoise, ynoise = 2 * np.random.rand() - 1, 2 * np.random.rand() - 1
                normz = np.hypot(xnoise, ynoise)
                xnoise, ynoise = xnoise * radius / normz, ynoise * radius / normz
                noisetrip[i, :] = TransferFunction.meters2lonlat(x + xnoise, y + ynoise)
        return noisetrip

    def downsampling(self, trip, rate):
        keep_idx = [0]
        for i in range(1, trip.shape[0] - 1):
            if np.random.rand() > rate:
                keep_idx.append(i)
        keep_idx.append(trip.shape[0] - 1)
        return trip[keep_idx, :]

    def downsamplingDistort(self, trip):
        """
        Downsampling + adding noise, only recalculate hot word encoding for points with downsampling and offset
        """
        noisetrips = []
        for dropping_rate in self.dropping_rates:
            noisetrip1 = self.downsampling(trip, dropping_rate)
            for distorting_rate in self.distorting_rates:
                noisetrip2 = self.distort(noisetrip1, distorting_rate)
                noisetrips.append(noisetrip2)
        return noisetrips


def resample_to_k_segments(trips, valid_lens, kseg):
    num_trips = len(trips)
    kseg_trips = []
    for trip, valid_len in zip(trips, valid_lens):
        ksegs = []

        trip = trip[:valid_len]
        seg = valid_len // kseg

        for i in range(kseg):
             if i == kseg - 1:
                 ksegs.append(np.mean(trip[i * seg:], axis=0))
             else:
                 ksegs.append(np.mean(trip[i * seg: i * seg + seg], axis=0))
        kseg_trips.append(ksegs)

    kseg_trips = np.array(kseg_trips)
    for feature_i in range(kseg_trips.shape[-1]):
        kseg_trips[..., feature_i] = (kseg_trips[..., feature_i] - np.min(kseg_trips[..., feature_i])) / \
                                     (np.max(kseg_trips[..., feature_i]) - np.min(kseg_trips[..., feature_i]))
    kseg_trips = kseg_trips.reshape(num_trips, -1)

    return kseg_trips


def main():
    parser = ArgumentParser()
    parser.add_argument('-n', '--name', help='the name of the dataset', type=str, required=True)
    parser.add_argument('-t', '--types', help='the type of meta data to dump', type=str, default="resample-60")
    parser.add_argument('-i', '--indices', help='the set index to dump meta', type=str, default="0,1,2")
    parser.add_argument('-g', '--grid', action='store_true', help="whether to project to grids.")

    args = parser.parse_args()

    road_type = 'grid' if args.grid else 'road_network'
    data = Data(args.name, road_type)
    data.read_hdf()
    for type in args.types.split(','):
        for i in args.indices.split(','):
            data.dump_meta(type, int(i))
            # Test if we can load meta from the file
            meta = data.load_meta(type, i)


if __name__ == '__main__':
    main()
