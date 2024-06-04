import torch
import os
import random
import numpy as np
from chinese_calendar import is_holiday

from common.trajectory import Trajectory, get_tid
from .parse_traj import ParseMMTraj_MTraj

class Dataset(torch.utils.data.Dataset):
    def __init__(self, trajs_dir, parameters):
        self.mbr = parameters.mbr
        self.grid_size = parameters.grid_size
        self.road_grid_size = parameters.road_grid_size
        self.time_span = parameters.time_span
        self.win_size = parameters.win_size
        self.ds_type = parameters.ds_type
        self.keep_ratio = parameters.keep_ratio
        
        self.src_grid_seqs, self.src_gps_seqs, self.src_pro_feas = [], [], []
        self.trg_gps_seqs, self.trg_eid_seqs, self.trg_rate_seqs = [], [], []
        
        self.src_eid_seqs, self.src_rate_seqs, self.src_road_index_seqs = [], [], []
        self.trg_tid_seqs, self.trg_index_seqs, self.trg_interpolated_grid_seqs, self.trg_interpolated_gps_seqs,  = [], [], [], []
        
        self.new_tid_seqs = []
        
        self.get_data(trajs_dir)
        
    def get_data(self, trajs_dir):
        """
        get dataset from trajectory files
        merge multiple seq (Trajectory()) list to seqs
        Args:
            trajs_dir (str): directory of trajectory files
        """
        parser = ParseMMTraj_MTraj()
        traj_paths = os.listdir(trajs_dir)
        
        for traj_file_name in traj_paths:
            trajs = parser.parse(os.path.join(trajs_dir, traj_file_name))
            
            for traj in trajs:
                ls_grid_seq_ls, ls_gps_seq_ls, feature_seq_ls, \
                trg_t_seq_ls, trg_index_seq_ls, trg_grid_seq_ls, trg_gps_seq_ls, \
                ls_eid_seq_ls, ls_rate_seq_ls, ls_road_index_seq_ls, \
                mm_gps_seq_ls, mm_eid_seq_ls, mm_rate_seq_ls, new_tid_seq = self.parse_traj(traj, self.win_size, self.ds_type, self.keep_ratio)
                
                if new_tid_seq is not None:
                    self.src_grid_seqs.extend(ls_grid_seq_ls)
                    self.src_gps_seqs.extend(ls_gps_seq_ls)
                    self.src_pro_feas.extend(feature_seq_ls)
                    
                    self.src_eid_seqs.extend(ls_eid_seq_ls)
                    self.src_rate_seqs.extend(ls_rate_seq_ls)
                    self.src_road_index_seqs.extend(ls_road_index_seq_ls)
                    
                    self.trg_gps_seqs.extend(mm_gps_seq_ls)
                    self.trg_eid_seqs.extend(mm_eid_seq_ls)
                    self.trg_rate_seqs.extend(mm_rate_seq_ls)
                    
                    self.trg_tid_seqs.extend(trg_t_seq_ls)
                    self.trg_index_seqs.extend(trg_index_seq_ls)
                    self.trg_interpolated_grid_seqs.extend(trg_grid_seq_ls)
                    self.trg_interpolated_gps_seqs.extend(trg_gps_seq_ls)
                    
                    self.new_tid_seqs.extend(new_tid_seq)
                    
    def parse_traj(self, traj, win_size, ds_type, keep_ratio):
        """
        parse a trajectory to get dataset
        the result is a list containing multiple trajectory sequences(xx_seq_ls)
        each trajectory sequence is parsed into source and target sequences
        Args:
        -----
        traj:
            Trajectory()
        win_size:
            window size of length for a single high sampling trajectory
        ds_type:
            ['uniform', 'random']
            uniform: sample GPS point every down_steps element.
                     the down_step is calculated by 1/remove_ratio
            random: randomly sample (1-down_ratio)*len(old_traj) points by ascending.
        keep_ratio:
            float. range in (0,1). The ratio that keep GPS points to total points.
        Returns:
            ls_grid_seq_ls: low sampling src grid sequence list
            ls_gps_seq_ls: low sampling src gps sequence list
            feature_seq_ls: low sampling src feature sequence list
            trg_t_seq_ls: target time sequence list
            trg_index_seq_ls: target index sequence list
            trg_grid_seq_ls: target grid sequence list
            trg_gps_seq_ls: target gps sequence list
            ls_eid_seq_ls: low sampling src eid sequence list
            ls_rate_seq_ls: low sampling src rate sequence list
            ls_road_index_seq_ls: low sampling src road index sequence list
            mm_gps_seq_ls: map matching result gps sequence list
            mm_eid_seq_ls: map matching result eid sequence list
            mm_rate_seq_ls: map matching result rate sequence list
            new_tid_seq: new tid sequence list
        --------
        
        """
        
        new_trajs = self.get_win_trajs(traj, win_size)
        
        # initialize the return variables
        new_tid_seq = []
        
        mm_gps_seq_ls, mm_eid_seq_ls, mm_rate_seq_ls = [], [], []
        ls_grid_seq_ls, ls_gps_seq_ls, feature_seq_ls = [], [], []
        
        trg_t_seq_ls, trg_index_seq_ls, trg_grid_seq_ls, trg_gps_seq_ls = [], [], [], []
        ls_eid_seq_ls, ls_rate_seq_ls, ls_road_index_seq_ls = [], [], []
        
        
        
        for traj in new_trajs:
            tmp_pt_seq = traj.pt_list
            new_tid_seq.append(traj.tid)
            
            # get target sequence(map matching result)
            mm_gps_seq, mm_eid_seq, mm_rate_seq = self.get_trg_seq(tmp_pt_seq)
            if mm_eid_seq is None:
                # if there is no map matching result, then return None, ignore this trajectory
                return None, None, None, None, None, None, None, None, None, None, None, None, None, None
            
            # get source sequence
            ds_pt_seq = self.downsample_traj(tmp_pt_seq, ds_type, keep_ratio)
            ls_grid_seq, ls_gps_seq, hours, ttl_t = self.get_src_seq(ds_pt_seq)
            feature_seq = self.get_pro_feature(ds_pt_seq, hours) 
            
            # get target info sequence(raw GPS points and their corresponding info)
            trg_t_seq, trg_index_seq, trg_grid_seq, trg_gps_seq = self.get_trg_info_seq(tmp_pt_seq, ds_pt_seq)
            
            # get source info sequence
            ls_eid_seq, ls_rate_seq, ls_road_index_seq = self.get_src_info_seq(ds_pt_seq)
            
            # check if the source sequence and target sequence have the same length, if not, assert False
            assert len(ls_grid_seq) == len(ls_gps_seq) == len(ls_eid_seq) == len(ls_rate_seq) 
            assert len(mm_gps_seq) == len(mm_eid_seq) == len(mm_rate_seq) == len(trg_grid_seq) == len(trg_gps_seq) == len(trg_t_seq) == len(trg_index_seq)
                 
            # append to the dataset
            mm_gps_seq_ls.append(mm_gps_seq)
            mm_eid_seq_ls.append(mm_eid_seq)
            mm_rate_seq_ls.append(mm_rate_seq)
            
            ls_grid_seq_ls.append(ls_grid_seq)
            ls_gps_seq_ls.append(ls_gps_seq)
            feature_seq_ls.append(feature_seq)
            
            trg_t_seq_ls.append(trg_t_seq)
            trg_index_seq_ls.append(trg_index_seq)
            trg_grid_seq_ls.append(trg_grid_seq)
            trg_gps_seq_ls.append(trg_gps_seq)
            
            ls_eid_seq_ls.append(ls_eid_seq)
            ls_rate_seq_ls.append(ls_rate_seq)
            ls_road_index_seq_ls.append(ls_road_index_seq)
            
        return ls_grid_seq_ls, ls_gps_seq_ls, feature_seq_ls, \
               trg_t_seq_ls, trg_index_seq_ls, trg_grid_seq_ls, trg_gps_seq_ls, \
               ls_eid_seq_ls, ls_rate_seq_ls, ls_road_index_seq_ls, \
               mm_gps_seq_ls, mm_eid_seq_ls, mm_rate_seq_ls, new_tid_seq
    
    def get_win_trajs(self, traj, win_size):
        pt_list = traj.pt_list
        len_pt_list = len(pt_list)
        if len_pt_list < win_size:
            return [traj]

        num_win = len_pt_list // win_size
        last_traj_len = len_pt_list % win_size + 1
        new_trajs = []
        for w in range(num_win):
            # if last window is large enough then split to a single trajectory
            if w == num_win - 1 and last_traj_len > 15:
                tmp_pt_list = pt_list[win_size * w - 1:]
            # elif last window is not large enough then merge to the last trajectory
            elif w == num_win - 1 and last_traj_len <= 15:
                # fix bug, when num_win = 1
                ind = 0
                if win_size * w - 1 > 0:
                    ind = win_size * w - 1
                tmp_pt_list = pt_list[ind:]
            # else split trajectories based on the window size
            else:
                tmp_pt_list = pt_list[max(0, (win_size * w - 1)):win_size * (w + 1)]
                # -1 to make sure the overlap between two trajs

            new_traj = Trajectory(traj.oid, get_tid(traj.oid, tmp_pt_list), tmp_pt_list)
            new_trajs.append(new_traj)
        return new_trajs
    
    def get_trg_seq(self, pt_list):
        mm_gps_seq, mm_eid_seq, mm_rate_seq = [], [], []
        for pt in pt_list:
            candi_pt = pt.data['candi_pt']
            if candi_pt is None:
                # if there is no map matching result, then return None
                return None, None, None
            else:
                mm_gps_seq.append([candi_pt.lat, candi_pt.lng])
                mm_eid_seq.append(candi_pt.eid)
                mm_rate_seq.append(candi_pt.rate)
                
        return mm_gps_seq, mm_eid_seq, mm_rate_seq
    
    def get_trg_info_seq(self, all_pt_seq, ds_pt_seq):
        trg_t_seq, trg_index_seq, trg_grid_seq, trg_gps_seq = [], [], [], []
        
        first_pt = all_pt_seq[0]
        time_interval = self.time_span
        
        for trg_pt in all_pt_seq:
            t = self.get_normalized_t(first_pt, trg_pt, time_interval)
            trg_t_seq.append(t)
            result = filter(lambda x: x.time == trg_pt.time, ds_pt_seq)
            result_seq = list(result)
            
            
            if len(result_seq) > 0:
                # if current point is in the downsampled point, then get its info
                trg_index_seq.append([1])
                candi_pt = trg_pt.data['candi_pt']
                grid_xid, grid_yid = self.gps2grid(candi_pt, self.mbr, self.grid_size)
                trg_grid_seq.append([grid_xid, grid_yid])
                trg_gps_seq.append([candi_pt.lat, candi_pt.lng])
            else:
                trg_index_seq.append([0])
                trg_grid_seq.append([0, 0])
                trg_gps_seq.append([0, 0])
                
            assert len(trg_t_seq) == len(trg_index_seq) == len(trg_grid_seq) == len(trg_gps_seq), 'The number of get_trg_grid_t must be equal.'
            
        for i in range(len(trg_index_seq)):
            if trg_index_seq[i] == [0]:
                # if current point is not in the downsampled point, then interpolate its info
                pre_i = i - 1
                next_i = i + 1
                while True:
                    if trg_index_seq[pre_i] == [1]:break
                    pre_i -= 1
                while True:
                    if trg_index_seq[next_i] == [1]:break
                    next_i += 1
                
                if pre_i < 0 or next_i >= len(trg_index_seq):
                    print(f"Warning: Skipping interpolation for index {i} due to boundary issues.")
                    continue
                
                all_interval, cur_interval = next_i - pre_i, i - pre_i
                all_lat, all_lng = trg_gps_seq[next_i][0] - trg_gps_seq[pre_i][0], trg_gps_seq[next_i][1] - trg_gps_seq[pre_i][1]
                start_lat, start_lng = trg_gps_seq[pre_i][0], trg_gps_seq[pre_i][1]
                cur_lat, cur_lng = all_lat / all_interval * cur_interval + start_lat, all_lng / all_interval * cur_interval + start_lng
                trg_gps_seq[i] = [cur_lat, cur_lng]
                grid_xid, grid_yid = self.gps2grid(pt=None, mbr=self.mbr, grid_size=self.grid_size, trg_new_grid=True, lat=cur_lat, lng=cur_lng)
                trg_grid_seq[i] = [grid_xid, grid_yid]
                
        return  trg_t_seq, trg_index_seq, trg_grid_seq, trg_gps_seq       
    
    def get_src_seq(self, pt_list):
        hours = []
        ls_grid_seq, ls_gps_seq = [], []
        
        first_pt, last_pt = pt_list[0], pt_list[-1]
        time_interval = self.time_span
        ttl_t = self.get_normalized_t(first_pt, last_pt, time_interval)
        
        for pt in pt_list:
            hours.append(pt.time.hour)
            t = self.get_normalized_t(first_pt, pt, time_interval)
            ls_gps_seq.append([pt.lat, pt.lng])
            
            grid_xid, grid_yid = self.gps2grid(pt, self.mbr, self.grid_size)
            ls_grid_seq.append([grid_xid, grid_yid, t])
            
        return ls_grid_seq, ls_gps_seq, hours, ttl_t
    
    def get_src_info_seq(self, pt_list):
        ls_eid_seq, ls_rate_seq, ls_road_index_seq = [], [], []
        for pt in pt_list:
            road_index_y, road_index_x = self.cal_index_y_x(pt.lat, pt.lng, self.mbr, interval=self.road_grid_size)
            ls_road_index_seq.append([pt.time.hour, road_index_y, road_index_x])
            
            if pt.data['candi_pt'] is not None:
                ls_eid_seq.append(pt.data['candi_pt'].eid)
                ls_rate_seq.append(pt.data['candi_pt'].rate)
            else:
                ls_eid_seq.append(0)
                ls_rate_seq.append(0)
        
        # if ls_eid_seq[i] == 0, then interpolate the eid and rate
        for i in range(len(ls_eid_seq)):
            if ls_eid_seq[i] != 0:
                continue
            else:
                pre_i = i - 1
                next_i = i + 1
            while True:
                if pre_i < 0 and next_i >= len(ls_eid_seq):
                    break
                
                if pre_i >= 0 and ls_eid_seq[pre_i] != 0:
                    ls_eid_seq[i] = ls_eid_seq[pre_i]
                    ls_rate_seq[i] = ls_rate_seq[pre_i]
                    break
                
                if next_i < len(ls_eid_seq) and ls_eid_seq[next_i] != 0:
                    ls_eid_seq[i] = ls_eid_seq[next_i]
                    ls_rate_seq[i] = ls_rate_seq[next_i]
                    break
                
                if pre_i >= 0:
                    pre_i -= 1
                if next_i < len(ls_eid_seq):
                    next_i += 1
        
        return ls_eid_seq, ls_rate_seq, ls_road_index_seq
    
    def get_pro_feature(self, pt_list, hours):
        """
        get pro feature for each point in the trajectory
        """
        holiday = is_holiday(pt_list[0].time) * 1.0
        hour = {'hour': np.bincount(hours).argmax()}  # find most frequent hours as hour of the trajectory
        features = self.one_hot(hour) + [holiday]
        return features
    
    def gps2grid(self, pt, mbr, grid_size, trg_new_grid=False, lat=0, lng = 0):
        """
        mbr:
            MBR class.
        grid size:
            int. in meter
        """
        LAT_PER_METER = 8.993203677616966e-06
        LNG_PER_METER = 1.1700193970443768e-05
        lat_unit = LAT_PER_METER * grid_size
        lng_unit = LNG_PER_METER * grid_size
        
        max_xid = int((mbr.max_lat - mbr.min_lat) / lat_unit) + 1
        max_yid = int((mbr.max_lng - mbr.min_lng) / lng_unit) + 1
        if trg_new_grid == False:
            lat = pt.lat
            lng = pt.lng
        locgrid_x = int((lat - mbr.min_lat) / lat_unit) + 1
        locgrid_y = int((lng - mbr.min_lng) / lng_unit) + 1
        
        if locgrid_x < 0: locgrid_x = 0
        if locgrid_x > max_xid: locgrid_x = max_xid
        if locgrid_y < 0: locgrid_y = 0
        if locgrid_y > max_yid: locgrid_y = max_yid

        return locgrid_x, locgrid_y
    
    def cal_index_y_x(self, lat, lng, mbr, interval):
        """
        calculate the index of road in y and x direction
        """
        start_lng = mbr.min_lng
        start_lat = mbr.min_lat
        end_lng = mbr.max_lng
        end_lat = mbr.max_lat
        
        lng_interval = abs(end_lng - start_lng) / interval   
        log_interval = abs(end_lat - start_lat) / interval
        if lng>=start_lng and lng < end_lng and lat>=start_lat and lat<end_lat:
            latitude=int(np.floor(abs(lat-start_lat) / log_interval))
            longitude=int(np.floor(abs(lng-start_lng) / lng_interval))
            return latitude, longitude
        else:
            return 0, 0
        
    def get_normalized_t(self, first_pt, current_pt, time_interval):
        """
        calculate normalized t from first and current pt
        return time index (normalized time)
        """
        t = int(1+((current_pt.time - first_pt.time).seconds/time_interval))
        return t
    
    @staticmethod
    def downsample_traj(pt_list, ds_type, keep_ratio):
        """
        Down sample trajectory
        Args:
        -----
        pt_list:
            list of Point()
        ds_type:
            ['uniform', 'random']
            uniform: sample GPS point every down_stepth element.
                     the down_step is calculated by 1/remove_ratio
            random: randomly sample (1-down_ratio)*len(old_traj) points by ascending.
        keep_ratio:
            float. range in (0,1). The ratio that keep GPS points to total points.
        Returns:
        -------
        traj:
            new Trajectory()
        """
        assert ds_type in ['uniform', 'random'], 'only `uniform` or `random` is supported'

        old_pt_list = pt_list.copy()
        start_pt = old_pt_list[0]
        end_pt = old_pt_list[-1]

        if ds_type == 'uniform':
            if (len(old_pt_list) - 1) % int(1 / keep_ratio) == 0:
                new_pt_list = old_pt_list[::int(1 / keep_ratio)]
            else:
                new_pt_list = old_pt_list[::int(1 / keep_ratio)] + [end_pt]
        elif ds_type == 'random':
            sampled_inds = sorted(
                random.sample(range(1, len(old_pt_list) - 1), int((len(old_pt_list) - 2) * keep_ratio)))
            new_pt_list = [start_pt] + list(np.array(old_pt_list)[sampled_inds]) + [end_pt]

        return new_pt_list
    
    @staticmethod
    def one_hot(data):
        one_hot_dict = {'hour': 24, 'weekday': 7, 'weather':5}
        for k, v in data.items():
            encoded_data = [0] * one_hot_dict[k]
            encoded_data[v - 1] = 1
        return encoded_data
    
    def add_token(self, sequence):
        """
        Append start element(sos in NLP) for each sequence. And convert each list to tensor.
        """
        new_sequence = []
        dimention = len(sequence[0])
        sos_token = [0] * dimention
        new_sequence.append(sos_token)
        new_sequence.extend(sequence)
        new_sequence = torch.tensor(new_sequence)
        return new_sequence
    
    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.src_grid_seqs)
    
    def __getitem__(self, index):
        """Generates one sample of data"""
        src_grid_seq = self.src_grid_seqs[index]
        src_gps_seq = self.src_gps_seqs[index]
        # src_pro_feas = self.src_pro_feas[index] a trajectory has only one pro feature
        
        src_eid_seq = self.src_eid_seqs[index]
        src_rate_seq = self.src_rate_seqs[index]
        src_road_index_seq = self.src_road_index_seqs[index]
        
        trg_gps_seq = self.trg_gps_seqs[index]
        trg_eid_seq = self.trg_eid_seqs[index]
        trg_rate_seq = self.trg_rate_seqs[index]
        
        
        trg_t_seq = self.trg_tid_seqs[index]
        trg_index_seq = self.trg_index_seqs[index]
        trg_interpolated_gps_seq = self.trg_interpolated_gps_seqs[index]
        trg_interpolated_grid_seq = self.trg_interpolated_grid_seqs[index]
        
        src_pro_feas = torch.tensor(self.src_pro_feas[index])
        src_grid_seq = self.add_token(src_grid_seq)
        src_gps_seq = self.add_token(src_gps_seq)
        src_eid_seq = self.add_token(src_eid_seq)
        src_rate_seq = self.add_token(src_rate_seq)
        src_road_index_seq = self.add_token(src_road_index_seq)
        
        trg_gps_seq = self.add_token(trg_gps_seq)
        trg_eid_seq = self.add_token(trg_eid_seq)
        trg_rate_seq = self.add_token(trg_rate_seq)
        
        trg_t_seq = self.add_token(trg_t_seq)
        trg_index_seq = self.add_token(trg_index_seq)
        trg_interpolated_gps_seq = self.add_token(trg_interpolated_gps_seq)
        trg_interpolated_grid_seq = self.add_token(trg_interpolated_grid_seq)
        
        return src_grid_seq, src_gps_seq, src_eid_seq, src_rate_seq, src_road_index_seq, src_pro_feas, \
               trg_gps_seq, trg_eid_seq, trg_rate_seq, trg_t_seq, trg_index_seq, trg_interpolated_gps_seq, trg_interpolated_grid_seq
        
        
def collate_fn(batch):
    """_summary_
    Reference: https://github.com/yunjey/seq2seq-dataloader/blob/master/data_loader.py
    Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Sequences are padded to the maximum length of mini-batch sequences (dynamic padding).
    Args:
    - batch: list of tuple (src_seq, trg_seq).
    Returns:
    - src_seq: tensor of shape (batch_size, src_seq_len, input_size).
    """
    
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        dim = sequences[0].size(1)
        padded_seqs = torch.zeros(len(sequences), max(lengths), dim).type_as(sequences[0]) 
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end, :] = seq[:end, :]
        return padded_seqs, lengths
    
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    
    # separate source and target sequences
    src_grid_seqs, src_gps_seqs, src_eid_seqs, src_rate_seqs, src_road_index_seqs, src_pro_feas, \
    trg_gps_seqs, trg_eid_seqs, trg_rate_seqs, trg_t_seqs, trg_index_seqs, trg_interpolated_gps_seqs, trg_interpolated_grid_seqs = zip(*batch)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_grid_seqs, src_lengths = merge(src_grid_seqs)
    src_gps_seqs, _ = merge(src_gps_seqs)
    src_eid_seqs, _ = merge(src_eid_seqs)
    src_rate_seqs, _ = merge(src_rate_seqs)
    src_road_index_seqs, _ = merge(src_road_index_seqs)
    src_pro_feas = torch.tensor([list(src_pro_fea) for src_pro_fea in src_pro_feas])

    trg_gps_seqs, trg_lengths = merge(trg_gps_seqs)
    trg_eid_seqs, _ = merge(trg_eid_seqs)
    trg_rate_seqs, _ = merge(trg_rate_seqs)
    trg_t_seqs, _ = merge(trg_t_seqs)
    trg_index_seqs, _ = merge(trg_index_seqs)
    trg_interpolated_gps_seqs, _ = merge(trg_interpolated_gps_seqs)
    trg_interpolated_grid_seqs, _ = merge(trg_interpolated_grid_seqs)
    
    return src_grid_seqs, src_gps_seqs, src_eid_seqs, src_rate_seqs, src_road_index_seqs, src_pro_feas, \
           trg_gps_seqs, trg_eid_seqs, trg_rate_seqs, trg_t_seqs, trg_index_seqs, trg_interpolated_gps_seqs, trg_interpolated_grid_seqs, \
           src_lengths, trg_lengths
    
