import time
import torch.utils
import torch.utils.data
from tqdm import tqdm
import logging
import sys
import argparse
import pandas as pd
import os
import torch
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import numpy as np
from utils.utils import save_json_data, create_dir, load_pkl_data
from common.mbr import MBR
from common.spatial_func import SPoint, distance
from common.road_network import load_rn_shp

from utils.datasets import Dataset, collate_fn
from models.model_utils import load_rn_dict, load_rid_freqs, get_rid_grid, get_poi_info, get_rn_info
from models.model_utils import get_online_info_dict, epoch_time, AttrDict, get_rid_rnfea_dict
from models.multi_train import evaluate, init_weights, train, pretrain, pre_evaluate
from models.demo2 import MM_STGED, DecoderMulti, Encoder, SharedEmbedding
from utils.utils import load_graph_adj_mtx, load_graph_node_features
import warnings
import json
from itertools import chain
warnings.filterwarnings("ignore", category=UserWarning)

import sys
sys.path.append('./')
sys.path.append('../')
"""
python main_demo2_pretrain.py --dataset Porto Chengdu --data_ratio 1 > demo2/pretrain_porto_chengdu.txt 2>&1 &
"""

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Multi-task Traj Interp')
    parser.add_argument('--dataset', nargs=2, default=['Porto', 'Chengdu'], help='data set')

    parser.add_argument('--num_emb', type=int, default=1024)
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--finetune_flag', default=False)

    parser.add_argument('--module_type', type=str, default='simple', help='module type')
    parser.add_argument('--keep_ratio', type=float, default=0.125, help='keep ratio in float')
    parser.add_argument('--lambda1', type=int, default=10, help='weight for multi task rate')
    parser.add_argument('--hid_dim', type=int, default=512, help='hidden dimension')
    parser.add_argument('--epochs', type=int, default=20, help='epochs')
    parser.add_argument('--grid_size', type=int, default=50, help='grid size in int')
    parser.add_argument('--dis_prob_mask_flag', type=bool, default=True, help='flag of using prob mask')
    
    parser.add_argument('--pro_features_flag', action='store_true', help='flag of using profile features')
    parser.add_argument('--online_features_flag', action='store_true', help='flag of using online features')
    parser.add_argument('--tandem_fea_flag', action='store_true', help='flag of using tandem rid features')
    
    parser.add_argument('--no_attn_flag', type=bool, default=True, help='flag of using attention')
    parser.add_argument('--load_pretrained_flag', default=False, help='flag of load pretrained model')
    parser.add_argument('--model_old_path', type=str, default='', help='old model path')
    parser.add_argument('--no_debug', type=bool, default=False, help='flag of debug')
    parser.add_argument('--no_train_flag', type=bool, default=True, help='flag of training')
    parser.add_argument('--test_flag', type=bool, default=True, help='flag of testing')
    parser.add_argument('--top_K', type=int, default=10, help='top K value in the decoder')
    parser.add_argument('--RD_inter', type=str, default='1h', help='路况的时间间隔')
    
    parser.add_argument('--data_ratio', type=float, default=0.1, help='the size ratio of used dataset')

    opts = parser.parse_args()

    debug = opts.no_debug
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

    args = [AttrDict() for _ in range(2)]
    spatial_A_trans_list = []
    road_condition_list = []
    SE_list = []
    train_iterator_list = []
    valid_iterator_list = []
    test_iterator_list = []
    rn_dict_list = []
    grid_rn_dict_list = []
    rn_list = []
    raw2new_rid_dict_list = []
    raw_rn_dict_list = []
    new2raw_rid_dict_list = []
    id_size_list = []
    for idx, city in enumerate(opts.dataset):
        if city == 'Porto':
            args_dict = {
                'module_type':opts.module_type,
                'debug':debug,
                'device':device,

                # pre train
                'load_pretrained_flag':opts.load_pretrained_flag,
                'model_old_path':opts.model_old_path,
                'train_flag':opts.no_train_flag,
                'test_flag':opts.test_flag,

                # attention
                'attn_flag':opts.no_attn_flag,

                # constranit
                'dis_prob_mask_flag':opts.dis_prob_mask_flag,
                'search_dist':50,
                'beta':15,

                # features
                'tandem_fea_flag':opts.tandem_fea_flag,
                'pro_features_flag':opts.pro_features_flag,
                'online_features_flag':opts.online_features_flag,

                # extra info module
                'rid_fea_dim':8,
                'pro_input_dim':25, # 24[hour] + 5[waether] + 1[holiday]  without weather
                'pro_output_dim':8,
                'poi_num':5,
                'online_dim':5+5,  # poi/roadnetwork features dim
                'poi_type':'company,food,shopping,viewpoint,house',
                'user_num': 439, 
                # MBR
                'min_lat':41.142,
                'min_lng':-8.652,
                'max_lat':41.174,
                'max_lng':-8.578,

                # input data params
                'keep_ratio':opts.keep_ratio,
                'grid_size':opts.grid_size,
                'time_span':15,
                'win_size':50,
                'ds_type':'uniform',
                'split_flag':False,
                'shuffle':True,
                'input_dim':3,

                # model params
                'hid_dim':opts.hid_dim,
                'id_emb_dim':128,
                'dropout':0.5,
                'id_size':1366+1,

                'lambda1':opts.lambda1,
                'n_epochs':opts.epochs,
                'top_K': opts.top_K,
                'batch_size':128,
                'learning_rate':1e-3,
                'tf_ratio':0.5,
                'clip':1,
                'log_step':1,

                'num_emb': opts.num_emb,
                'emb_dim': opts.emb_dim
            }
        elif city == 'Chengdu':
            args_dict = {
                'module_type':opts.module_type,
                'debug':debug,
                'device':device,

                # pre train
                'load_pretrained_flag':opts.load_pretrained_flag,
                'model_old_path':opts.model_old_path,
                'train_flag':opts.no_train_flag,
                'test_flag':opts.test_flag,

                # attention
                'attn_flag':opts.no_attn_flag,

                # constranit
                'dis_prob_mask_flag':opts.dis_prob_mask_flag,
                'search_dist':50,
                'beta':15,

                # features
                'tandem_fea_flag':opts.tandem_fea_flag,
                'pro_features_flag':opts.pro_features_flag,
                'online_features_flag':opts.online_features_flag,

                # extra info module
                'rid_fea_dim':8,
                'pro_input_dim':25, # 24[hour] + 5[waether] + 1[holiday]  without weather
                'pro_output_dim':8,
                'poi_num':5,
                'online_dim':5+5,  # poi/roadnetwork features dim
                'poi_type':'company,food,shopping,viewpoint,house',
                # 'user_num': 77499 ,# 17675,
                'user_num': 62075, #TODO: change this line!

                # MBR
                'min_lat':30.655,
                'min_lng':104.043,
                'max_lat':30.727,
                'max_lng':104.129,

                # input data params
                'keep_ratio':opts.keep_ratio,
                'grid_size':opts.grid_size,
                'time_span':15,
                'win_size':50,
                'ds_type':'uniform',
                'split_flag':False,
                'shuffle':True,
                'input_dim':3,

                # model params
                'hid_dim':opts.hid_dim,
                'id_emb_dim':128,
                'dropout':0.5,
                'id_size':2902+1,# 2504+1,

                'lambda1':opts.lambda1,
                'n_epochs':opts.epochs,
                'top_K': opts.top_K,
                'RD_inter': opts.RD_inter,
                'batch_size':128,
                'learning_rate':1e-3,
                'tf_ratio':0.5,
                'clip':1,
                'log_step':1,

                'num_emb': opts.num_emb,
                'emb_dim': opts.emb_dim
            }
        elif city == 'Xian':
            args_dict = {
                'module_type':opts.module_type,
                'debug':debug,
                'device':device,

                # pre train
                'load_pretrained_flag':opts.load_pretrained_flag,
                'model_old_path':opts.model_old_path,
                'train_flag':opts.no_train_flag,
                'test_flag':opts.test_flag,

                # attention
                'attn_flag':opts.no_attn_flag,

                # constranit
                'dis_prob_mask_flag':opts.dis_prob_mask_flag,
                'search_dist':50,
                'beta':15,

                # features
                'tandem_fea_flag':opts.tandem_fea_flag,
                'pro_features_flag':opts.pro_features_flag,
                'online_features_flag':opts.online_features_flag,

                # extra info module
                'rid_fea_dim':8,
                'pro_input_dim':25, # 24[hour] + 5[waether] + 1[holiday]  without weather
                'pro_output_dim':8,
                'poi_num':5,
                'online_dim':5+5,  # poi/roadnetwork features dim
                'poi_type':'company,food,shopping,viewpoint,house',
                'user_num': 32135 ,# 17675,

                # MBR
                'min_lat':34.2,
                'min_lng':108.92,
                'max_lat':34.28,
                'max_lng':109.01,

                # input data params
                'keep_ratio':opts.keep_ratio,
                'grid_size':opts.grid_size,
                'time_span':15,
                'win_size':50,
                'ds_type':'uniform',
                'split_flag':False,
                'shuffle':True,
                'input_dim':3,

                # model params
                'hid_dim':opts.hid_dim,
                'id_emb_dim':128,
                'dropout':0.5,
                'id_size':1964+1,

                'lambda1':opts.lambda1,
                'n_epochs':opts.epochs,
                'top_K': opts.top_K,
                'RD_inter': opts.RD_inter,
                'batch_size':128,
                'learning_rate':1e-3,
                'tf_ratio':0.5,
                'clip':1,
                'log_step':1,

                'num_emb': opts.num_emb,
                'emb_dim': opts.emb_dim
            }
    
        assert city in ['Porto', 'Chengdu', 'Xian'], 'Check dataset name if in [Porto, Chengdu, Xian]'
        args_dict['data_ratio'] = opts.data_ratio
        args[idx].update(args_dict)

        print('Preparing data...')

        train_trajs_dir = "/workspace/guozuyu/final_data/{}/train_data/".format(city)
        valid_trajs_dir = "/workspace/guozuyu/final_data/{}/valid_data/".format(city)
        test_trajs_dir = "/workspace/guozuyu/final_data/{}/test_data/".format(city)

        extra_info_dir = "/workspace/guozuyu/map/{}/extra_data/".format(city)
        rn_dir = "/workspace/guozuyu/map/{}/".format(city)
        user_dir = json.load(open( extra_info_dir + "uid2index.json"))
        SE_file = extra_info_dir + '{}_SE_128.txt'.format(city)
        condition_file = extra_info_dir + 'flow_new.npy'
        road_file = extra_info_dir + 'graph_A.csv'

        if args[idx].tandem_fea_flag:
            fea_flag = True
        else:
            fea_flag = False
        
        #TODO: macro trajectory flow graph
        spatial_A = load_graph_adj_mtx(road_file)
        spatial_A_trans = np.zeros((spatial_A.shape[0]+1, spatial_A.shape[1]+1)) + 1e-10
        spatial_A_trans[1:,1:] = spatial_A  # (# of road segments + 1, # of segments + 1)

        road_condition = np.load(condition_file) # T, N, N (# of grid row, # of grid col, T, feature_size)
        for i in range(road_condition.shape[0]):
            maxn = road_condition[i].max()
            road_condition[i] = road_condition[i] / maxn

        f = open(SE_file, mode = 'r')
        lines = f.readlines()
        temp = lines[0].split(' ')
        N, dims = int(temp[0])+1, int(temp[1])  # num of road segments + 1, feature size
        SE = np.zeros(shape = (N, dims), dtype = np.float32)
        for line in lines[1 :]:
            temp = line.split(' ')
            index = int(temp[0])
            SE[index+1] = temp[1 :]
    
        SE = torch.from_numpy(SE)

        rn = load_rn_shp(rn_dir, is_directed=True)
        raw_rn_dict = load_rn_dict(extra_info_dir, file_name='raw_rn_dict.json')         # query road segment infomation (coords, length, level) by raw rid(str)
        new2raw_rid_dict = load_rid_freqs(extra_info_dir, file_name='new2raw_rid.json')  # query raw rid by new rid
        raw2new_rid_dict = load_rid_freqs(extra_info_dir, file_name='raw2new_rid.json')  # query new rid by raw rid
        rn_dict = load_rn_dict(extra_info_dir, file_name='rn_dict.json')                 # query road segment infomation by rid(int)
        mbr = MBR(args[idx].min_lat, args[idx].min_lng, args[idx].max_lat, args[idx].max_lng)
        grid_rn_dict, max_xid, max_yid = get_rid_grid(mbr, args[idx].grid_size, rn_dict)      # query [rid(s)] by grid index
        args_dict['max_xid'] = max_xid
        args_dict['max_yid'] = max_yid
        args[idx].update(args_dict)
        print(args)
    
    # load features
        weather_dict = None 
        if args[idx].online_features_flag:
            grid_poi_df = pd.read_csv(extra_info_dir+'poi'+str(args.grid_size)+'.csv',index_col=[0,1])
            norm_grid_poi_dict = get_poi_info(grid_poi_df, args)
            norm_grid_rnfea_dict = get_rn_info(rn, mbr, args.grid_size, grid_rn_dict, rn_dict)
            online_features_dict = get_online_info_dict(grid_rn_dict, norm_grid_poi_dict, norm_grid_rnfea_dict, args)
        else:
            norm_grid_poi_dict, norm_grid_rnfea_dict, online_features_dict = None, None, None
        rid_features_dict = None

        spatial_A_trans_list.append(spatial_A_trans)
        road_condition_list.append(road_condition)
        SE_list.append(SE)
        rn_dict_list.append(rn_dict)
        grid_rn_dict_list.append(grid_rn_dict)
        rn_list.append(rn)
        raw2new_rid_dict_list.append(raw2new_rid_dict)
        raw_rn_dict_list.append(raw_rn_dict)
        new2raw_rid_dict_list.append(new2raw_rid_dict)
        id_size_list.append(args_dict['id_size'])

        train_dataset = Dataset(train_trajs_dir, user_dir, raw2new_rid_dict_list[idx], mbr=mbr, norm_grid_poi_dict=norm_grid_poi_dict,
                                norm_grid_rnfea_dict=norm_grid_rnfea_dict, weather_dict=weather_dict,
                                parameters=args[idx], debug=debug)  # default arguments are set to None
        valid_dataset = Dataset(valid_trajs_dir, user_dir, raw2new_rid_dict_list[idx], mbr=mbr, norm_grid_poi_dict=norm_grid_poi_dict,
                                norm_grid_rnfea_dict=norm_grid_rnfea_dict, weather_dict=weather_dict,
                                parameters=args[idx], debug=debug)
        test_dataset = Dataset(test_trajs_dir, user_dir, raw2new_rid_dict_list[idx], mbr=mbr, norm_grid_poi_dict=norm_grid_poi_dict,
                                norm_grid_rnfea_dict=norm_grid_rnfea_dict, weather_dict=weather_dict,
                                parameters=args[idx], debug=debug)

        train_subset_indices = torch.randperm(int(len(train_dataset) * opts.data_ratio)).tolist()
        valid_subset_indices = torch.randperm(int(len(valid_dataset) * opts.data_ratio)).tolist()
        test_subset_indices = torch.randperm(int(len(test_dataset) * opts.data_ratio)).tolist()

        train_sampler = torch.utils.data.SubsetRandomSampler(train_subset_indices)
        valid_sampler = torch.utils.data.SubsetRandomSampler(valid_subset_indices)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_subset_indices)

        train_iterator = torch.utils.data.DataLoader(train_dataset, batch_size=args[idx].batch_size, collate_fn=collate_fn,
                                                    num_workers=4, pin_memory=False, sampler=train_sampler)
        valid_iterator = torch.utils.data.DataLoader(valid_dataset, batch_size=args[idx].batch_size, collate_fn=collate_fn,
                                                    num_workers=4, pin_memory=False, sampler=valid_sampler)
        test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=args[idx].batch_size, collate_fn=collate_fn,
                                                num_workers=4, pin_memory=False, sampler=test_sampler)

        train_iterator_list.append(train_iterator)
        valid_iterator_list.append(valid_iterator)
        test_iterator_list.append(test_iterator)

    model_save_path = './demo2/'+time.strftime("%Y%m%d_%H%M%S") + '/'
    create_dir(model_save_path)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename=model_save_path + 'log.txt',
                        filemode='a+')

    enc = Encoder(args[0])
    dec = DecoderMulti(args[0])
    shared_emb = SharedEmbedding(args[0])
    model1 = MM_STGED(enc, dec, shared_emb, args[0].num_emb, args[0].id_size, args[0].hid_dim, args[0].id_emb_dim, args[0].hid_dim, args[0].max_xid, args[0].max_yid, args[0].top_K).to(device)
    model1.apply(init_weights)  # learn how to init weights

    model2 = MM_STGED(enc, dec, shared_emb, args[1].num_emb, args[1].id_size, args[1].hid_dim, args[1].id_emb_dim, args[1].hid_dim, args[1].max_xid, args[1].max_yid, args[1].top_K).to(device)
    model2.apply(init_weights)

    if args[0].load_pretrained_flag:
        model1.load_state_dict(torch.load(args[0].model_old_path + 'val-best-model.pt'))
    
    # logging.info('model' + str(model1))
    # with open(model_save_path+'logging.txt', 'a+') as f:
        # f.write('model' + str(model1) + '\n')

    writer = SummaryWriter(log_dir='demo2')

    if args[0].train_flag:
        ls_train_loss, ls_train_id_acc1, ls_train_id_recall, ls_train_id_precision, \
        ls_train_rate_loss, ls_train_id_loss = [], [], [], [], [], []
        ls_valid_loss, ls_valid_id_acc1, ls_valid_id_recall, ls_valid_id_precision, \
        ls_valid_dis_mae_loss, ls_valid_dis_rmse_loss = [], [], [], [], [], []
        ls_valid_dis_rn_mae_loss, ls_valid_dis_rn_rmse_loss, ls_valid_rate_loss, ls_valid_id_loss = [], [], [], []

        dict_train_loss = {}
        dict_valid_loss = {}
        best_valid_loss = float('inf')  # compare id loss

        # get all parameters (model parameters + task dependent log variances)
        log_vars = [torch.zeros((1,), requires_grad=True, device=device)] * 2  # use for auto-tune multi-task param
        optimizer = optim.AdamW(chain(model1.parameters(), model2.parameters()), lr=args[0].learning_rate)
        for epoch in tqdm(range(args[0].n_epochs)):
            start_time = time.time()
            print("epoch:{}\n".format(epoch))
            print("start training.")
            new_log_vars, train_loss, train_id_acc1, train_id_recall, train_id_precision, \
            train_rate_loss, train_id_loss = pretrain(model1, model2, spatial_A_trans_list, road_condition_list, SE_list, train_iterator_list, optimizer, log_vars,
                                                   rn_dict_list, grid_rn_dict_list, rn_list, raw2new_rid_dict_list,
                                                   id_size_list, args)
            print("training time: ",time.time() - start_time)
            valid_id_acc1, valid_id_recall, valid_id_precision, valid_dis_mae_loss, valid_dis_rmse_loss, \
            valid_dis_rn_mae_loss, valid_dis_rn_rmse_loss, \
            valid_rate_loss, valid_id_loss = pre_evaluate(model1, model2, spatial_A_trans_list, road_condition_list, SE_list, valid_iterator_list,
                                                      rn_dict_list, grid_rn_dict_list, rn_list, raw2new_rid_dict_list,
                                                      online_features_dict, rid_features_dict, raw_rn_dict_list,
                                                      new2raw_rid_dict_list, args)
            ls_train_loss.append(train_loss)
            ls_train_id_acc1.append(train_id_acc1)
            ls_train_id_recall.append(train_id_recall)
            ls_train_id_precision.append(train_id_precision)
            ls_train_rate_loss.append(train_rate_loss)
            ls_train_id_loss.append(train_id_loss)

            ls_valid_id_acc1.append(valid_id_acc1)
            ls_valid_id_recall.append(valid_id_recall)
            ls_valid_id_precision.append(valid_id_precision)
            ls_valid_dis_mae_loss.append(valid_dis_mae_loss)
            ls_valid_dis_rmse_loss.append(valid_dis_rmse_loss)
            ls_valid_dis_rn_mae_loss.append(valid_dis_rn_mae_loss)
            ls_valid_dis_rn_rmse_loss.append(valid_dis_rn_rmse_loss)
            ls_valid_rate_loss.append(valid_rate_loss)
            ls_valid_id_loss.append(valid_id_loss)
            valid_loss = valid_rate_loss + valid_id_loss
            ls_valid_loss.append(valid_loss)

            dict_train_loss['train_ttl_loss'] = ls_train_loss
            dict_train_loss['train_id_acc1'] = ls_train_id_acc1
            dict_train_loss['train_id_recall'] = ls_train_id_recall
            dict_train_loss['train_id_precision'] = ls_train_id_precision
            dict_train_loss['train_rate_loss'] = ls_train_rate_loss
            dict_train_loss['train_id_loss'] = ls_train_id_loss

            dict_valid_loss['valid_ttl_loss'] = ls_valid_loss
            dict_valid_loss['valid_id_acc1'] = ls_valid_id_acc1
            dict_valid_loss['valid_id_recall'] = ls_valid_id_recall
            dict_valid_loss['valid_id_precision'] = ls_valid_id_precision
            dict_valid_loss['valid_rate_loss'] = ls_valid_rate_loss
            dict_valid_loss['valid_dis_mae_loss'] = ls_valid_dis_mae_loss
            dict_valid_loss['valid_dis_rmse_loss'] = ls_valid_dis_rmse_loss
            dict_valid_loss['valid_dis_rn_mae_loss'] = ls_valid_dis_rn_mae_loss
            dict_valid_loss['valid_dis_rn_rmse_loss'] = ls_valid_dis_rn_rmse_loss
            dict_valid_loss['valid_id_loss'] = ls_valid_id_loss

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model1.state_dict(), model_save_path + 'val-best-model1.pt')
                torch.save(model2.state_dict(), model_save_path + 'val-best-model2.pt')
                torch.save(enc.state_dict(), model_save_path + 'enc-best-model.pt')
                torch.save(shared_emb.state_dict(), model_save_path + 'emb-best-model.pt')
                torch.save(dec.state_dict(), model_save_path + 'dec-best-model.pt')

            if (epoch % args[0].log_step == 0) or (epoch == args[0].n_epochs - 1):
                logging.info('Epoch: ' + str(epoch + 1) + ' Time: ' + str(epoch_mins) + 'm' + str(epoch_secs) + 's')
                weights = [torch.exp(weight) ** 0.5 for weight in new_log_vars]
                logging.info('log_vars:' + str(weights))
                logging.info('\tTrain Loss:' + str(train_loss) +
                             '\tTrain RID Acc1:' + str(train_id_acc1) +
                             '\tTrain RID Recall:' + str(train_id_recall) +
                             '\tTrain RID Precision:' + str(train_id_precision) +
                             '\tTrain Rate Loss:' + str(train_rate_loss) +
                             '\tTrain RID Loss:' + str(train_id_loss))
                logging.info('\tValid Loss:' + str(valid_loss) +
                             '\tValid RID Acc1:' + str(valid_id_acc1) +
                             '\tValid RID Recall:' + str(valid_id_recall) +
                             '\tValid RID Precision:' + str(valid_id_precision) +
                             '\tValid Distance MAE Loss:' + str(valid_dis_mae_loss) +
                             '\tValid Distance RMSE Loss:' + str(valid_dis_rmse_loss) +
                             '\tValid Distance RN MAE Loss:' + str(valid_dis_rn_mae_loss) +
                             '\tValid Distance RN RMSE Loss:' + str(valid_dis_rn_rmse_loss) +
                             '\tValid Rate Loss:' + str(valid_rate_loss) +
                             '\tValid RID Loss:' + str(valid_id_loss))
                with open(model_save_path+'logging.txt', 'a+') as f:
                    f.write('Epoch: ' + str(epoch + 1) + ' Time: ' + str(epoch_mins) + 'm' + str(epoch_secs) + 's' + '\n')
                    f.write('\tTrain Loss:' + str(train_loss) +
                             '\tTrain RID Acc1:' + str(train_id_acc1) +
                             '\tTrain RID Recall:' + str(train_id_recall) +
                             '\tTrain RID Precision:' + str(train_id_precision) +
                             '\tTrain Rate Loss:' + str(train_rate_loss) +
                             '\tTrain RID Loss:' + str(train_id_loss) + 
                             '\n')
                    f.write('\tValid Loss:' + str(valid_loss) +
                             '\tValid RID Acc1:' + str(valid_id_acc1) +
                             '\tValid RID Recall:' + str(valid_id_recall) +
                             '\tValid RID Precision:' + str(valid_id_precision) +
                             '\tValid Distance MAE Loss:' + str(valid_dis_mae_loss) +
                             '\tValid Distance RMSE Loss:' + str(valid_dis_rmse_loss) +
                             '\tValid Distance RN MAE Loss:' + str(valid_dis_rn_mae_loss) +
                             '\tValid Distance RN RMSE Loss:' + str(valid_dis_rn_rmse_loss) +
                             '\tValid Rate Loss:' + str(valid_rate_loss) +
                             '\tValid RID Loss:' + str(valid_id_loss) + 
                             '\n')
                    f.write('\n')
                torch.save(model1.state_dict(), model_save_path + 'train-mid-model1.pt')
                torch.save(model2.state_dict(), model_save_path + 'train-mid-model2.pt')
                save_json_data(dict_train_loss, model_save_path, "train_loss.json")
                save_json_data(dict_valid_loss, model_save_path, "valid_loss.json")

                # Train
                writer.add_scalar("Loss/Train Loss", train_loss, epoch)
                writer.add_scalar("Metric/Train RID Acc", train_id_acc1, epoch)
                writer.add_scalar("Metric/Train RID Recall", train_id_recall, epoch)
                writer.add_scalar("Metric/Train RID Precision", train_id_precision, epoch)
                writer.add_scalar("Loss/Train Rate Loss", train_rate_loss, epoch)
                writer.add_scalar("Loss/Train RID Loss", train_id_loss, epoch)

                # Validate
                writer.add_scalar("Loss/Valid Loss", valid_loss, epoch)
                writer.add_scalar("Metric/Valid RID Acc", valid_id_acc1, epoch)
                writer.add_scalar("Metric/Valid RID Recall", valid_id_recall, epoch)
                writer.add_scalar("Metric/Valid RID Precision", valid_id_precision, epoch)
                writer.add_scalar("Loss/Valid Distance MAE Loss", valid_dis_mae_loss, epoch)
                writer.add_scalar("Loss/Valid Distance RMSE Loss", valid_dis_rmse_loss, epoch)
                writer.add_scalar("Loss/Valid Rate Loss", valid_rate_loss, epoch)
                writer.add_scalar("Loss/Valid RID Loss", valid_id_loss, epoch)
            
            writer.flush()

    if args[0].test_flag:
        model1.load_state_dict(torch.load(model_save_path + 'val-best-model1.pt'))
        model2.load_state_dict(torch.load(model_save_path + 'val-best-model2.pt'))
        start_time = time.time()
        models = [model1, model2]
        for i, model in enumerate(models):
            test_id_acc1, test_id_recall, test_id_precision, test_dis_mae_loss, test_dis_rmse_loss, \
            test_dis_rn_mae_loss, test_dis_rn_rmse_loss, test_rate_loss, test_id_loss = evaluate(model, spatial_A_trans_list[i], road_condition_list[i], SE_list[i], test_iterator_list[i],
                                                                                                rn_dict_list[i], grid_rn_dict_list[i], rn_list[i],
                                                                                                raw2new_rid_dict_list[i],
                                                                                                online_features_dict,
                                                                                                rid_features_dict,
                                                                                                raw_rn_dict_list[i], new2raw_rid_dict_list[i],
                                                                                                args[i])
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            logging.info('Test Time: ' + str(epoch_mins) + 'm' + str(epoch_secs) + 's')
            logging.info('\tTest RID Acc1:' + str(test_id_acc1) +
                        '\tTest RID Recall:' + str(test_id_recall) +
                        '\tTest RID Precision:' + str(test_id_precision) +
                        '\tTest Distance MAE Loss:' + str(test_dis_mae_loss) +
                        '\tTest Distance RMSE Loss:' + str(test_dis_rmse_loss) +
                        '\tTest Distance RN MAE Loss:' + str(test_dis_rn_mae_loss) +
                        '\tTest Distance RN RMSE Loss:' + str(test_dis_rn_rmse_loss) +
                        '\tTest Rate Loss:' + str(test_rate_loss) +
                        '\tTest RID Loss:' + str(test_id_loss))
            
            with open(model_save_path+'logging.txt', 'a+') as f:
                f.write("\n")
                f.write('Test Time: ' + str(epoch_mins) + 'm' + str(epoch_secs) + 's' + '\n')
                f.write('\tTest RID Acc1:' + str(test_id_acc1) +
                        '\tTest RID Recall:' + str(test_id_recall) +
                        '\tTest RID Precision:' + str(test_id_precision) +
                        '\tTest Distance MAE Loss:' + str(test_dis_mae_loss) +
                        '\tTest Distance RMSE Loss:' + str(test_dis_rmse_loss) +
                        '\tTest Distance RN MAE Loss:' + str(test_dis_rn_mae_loss) +
                        '\tTest Distance RN RMSE Loss:' + str(test_dis_rn_rmse_loss) +
                        '\tTest Rate Loss:' + str(test_rate_loss) +
                        '\tTest RID Loss:' + str(test_id_loss) +
                        '\n')