import numpy as np
import random

import torch
import torch.nn as nn

from models.model_utils import toseq, get_constraint_mask
from models.loss_fn import cal_id_acc_batch, cal_diss_loss_batch
from models.trajectory_graph import build_graph,search_road_index

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('multi_task device', device)


def init_weights(self):
    """
    Here we reproduce Keras default initialization weights for consistency with Keras version
    Reference: https://github.com/vonfeng/DeepMove/blob/master/codes/model.py
    """
    ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
    hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
    b = (param.data for name, param in self.named_parameters() if 'bias' in name)

    for t in ih:
        nn.init.xavier_uniform_(t)
    for t in hh:
        nn.init.orthogonal_(t)
    for t in b:
        nn.init.constant_(t, 0)


def train(model, spatial_A_trans, road_condition, SE, iterator, optimizer, log_vars, rn_dict, grid_rn_dict, rn,
          raw2new_rid_dict, online_features_dict, rid_features_dict, parameters):
    model.train()  # not necessary to have this line but it's safe to use model.train() to train model

    criterion_reg = nn.MSELoss()
    criterion_ce = nn.NLLLoss()

    epoch_ttl_loss = 0
    epoch_id1_loss = 0
    epoch_recall_loss = 0
    epoch_precision_loss = 0
    epoch_train_id_loss = 0
    epoch_rate_loss = 0
    for i, batch in enumerate(iterator):
        print("batch:{}".format(i))
        import time
        
        src_grid_seqs, src_gps_seqs, src_road_index_seqs, src_eid_seqs, src_rate_seqs, trg_in_t_seqs, trg_in_index_seqs, trg_in_grid_seqs, trg_in_gps_seqs, \
                    src_pro_feas, src_lengths, trg_gps_seqs, trg_rids, trg_rates, trg_lengths, use_id_seq = batch
        
        if parameters.dis_prob_mask_flag:
            constraint_mat, pre_grids, next_grids = get_constraint_mask(src_grid_seqs, src_gps_seqs, src_lengths,
                                                                        trg_lengths, grid_rn_dict, rn, raw2new_rid_dict,
                                                                        parameters)
            constraint_mat = constraint_mat.permute(1, 0, 2).to(device)
            pre_grids = pre_grids.permute(1, 0, 2).to(device)
            next_grids = next_grids.permute(1, 0, 2).to(device)
            # new_constraint_list = new_constraint_list.permute(1, 0, 2).to(device)
        else:
            max_trg_len = max(trg_lengths)
            batch_size = src_grid_seqs.size(0)
            constraint_mat = torch.zeros(max_trg_len, batch_size, parameters.id_size, device=device)
            pre_grids = torch.zeros(max_trg_len, batch_size, 3).to(device)
            next_grids = torch.zeros(max_trg_len, batch_size, 3).to(device)
            
        SE = SE.to(device)
        src_grid_seqs = src_grid_seqs.to(device) # [batch size, src len, 2]
        src_eid_seqs = src_eid_seqs.to(device)
        src_rate_seqs = src_rate_seqs.to(device)
        src_road_index_seqs = src_road_index_seqs.long().to(device)
        trg_in_grid_seqs = trg_in_grid_seqs.to(device)
        trg_rids = trg_rids.permute(1, 0, 2).long().to(device)
        trg_rates = trg_rates.permute(1, 0, 2).to(device)
        trg_in_index_seqs = trg_in_index_seqs.to(device)
        
        road_condition = torch.tensor(road_condition, dtype=torch.float).to(device)
        tra_time_A, tra_loca_A = build_graph(src_lengths, src_grid_seqs, src_gps_seqs)
        tra_time_A, tra_loca_A = tra_time_A.to(device), tra_loca_A.to(device)
        spatial_A_trans = torch.tensor(spatial_A_trans, dtype=torch.float).to(device)
        
        trg_gps_seqs = trg_gps_seqs.permute(1, 0, 2).to(device) # [trg len, batch size, 2]
        
        use_id_seq = torch.tensor(np.array(use_id_seq), dtype=torch.long).to(device)
        use_id_seq = use_id_seq.unsqueeze(0)
        
        start_time = time.time()
        optimizer.zero_grad()
        output_ids, output_rates = model(use_id_seq, spatial_A_trans, road_condition, src_road_index_seqs, SE, tra_time_A, tra_loca_A, src_lengths, src_grid_seqs, src_eid_seqs, src_rate_seqs, trg_in_grid_seqs, 
                    trg_in_index_seqs, trg_rids, trg_rates, trg_lengths, pre_grids, next_grids, constraint_mat,
                                             src_pro_feas, online_features_dict, rid_features_dict,
                                             teacher_forcing_ratio=0)
        
        
        output_rates = output_rates.squeeze(2)
        trg_rids = trg_rids.squeeze(2)
        
        trg_rates = trg_rates.squeeze(2)
        
        trg_lengths_sub = [length - 1 for length in trg_lengths]
        loss_ids1, recall, precision = cal_id_acc_batch(output_ids[1:], trg_rids[1:], trg_lengths_sub)

        # for bbp
        output_ids_dim = output_ids.shape[-1]
        output_ids = output_ids[1:].reshape(-1, output_ids_dim)  # [(trg len - 1)* batch size, output id one hot dim]
        trg_rids = trg_rids[1:].reshape(-1)  # [(trg len - 1) * batch size],
        
        loss_train_ids = criterion_ce(output_ids, trg_rids)
        loss_rates = criterion_reg(output_rates[1:], trg_rates[1:]) * parameters.lambda1
        ttl_loss = loss_train_ids + loss_rates
        
        
        ttl_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), parameters.clip)  # log_vars are not necessary to clip
        optimizer.step()
        print("train time:{}".format(time.time()-start_time))
        
        
        epoch_ttl_loss += ttl_loss.item()
        epoch_id1_loss += loss_ids1
        epoch_recall_loss += recall
        epoch_precision_loss += precision
        epoch_train_id_loss += loss_train_ids.item()
        epoch_rate_loss += loss_rates.item()
    # exit()
    return log_vars, epoch_ttl_loss / len(iterator), epoch_id1_loss / len(iterator), epoch_recall_loss / len(iterator), \
           epoch_precision_loss / len(iterator), epoch_rate_loss / len(iterator), epoch_train_id_loss / len(iterator)

def test(model, spatial_A_trans, road_condition, SE, iterator, rn_dict, grid_rn_dict, rn, raw2new_rid_dict,
             online_features_dict, rid_features_dict, raw_rn_dict, new2raw_rid_dict, parameters):
    model.eval()  # must have this line since it will affect dropout and batch normalization
    
    epoch_dis_mae_loss = 0
    epoch_dis_rmse_loss = 0
    epoch_dis_rn_mae_loss = 0
    epoch_dis_rn_rmse_loss = 0
    epoch_id1_loss = 0
    epoch_recall_loss = 0
    epoch_precision_loss = 0
    epoch_rate_loss = 0
    epoch_id_loss = 0 # loss from dl model
    criterion_ce = nn.NLLLoss()
    criterion_reg = nn.MSELoss()

    with torch.no_grad():  # this line can help speed up evaluation
        for i, batch in enumerate(iterator):
            
            src_grid_seqs, src_gps_seqs, src_road_index_seqs, src_eid_seqs, src_rate_seqs, trg_in_t_seqs, trg_in_index_seqs, trg_in_grid_seqs, trg_in_gps_seqs, \
                    src_pro_feas, src_lengths, trg_gps_seqs, trg_rids, trg_rates, trg_lengths, use_id_seq = batch

            if parameters.dis_prob_mask_flag:
                constraint_mat, pre_grids, next_grids = get_constraint_mask(src_grid_seqs, src_gps_seqs, src_lengths,
                                                                            trg_lengths, grid_rn_dict, rn,
                                                                            raw2new_rid_dict, parameters)
                constraint_mat = constraint_mat.permute(1, 0, 2).to(device)
                pre_grids = pre_grids.permute(1, 0, 2).to(device)
                next_grids = next_grids.permute(1, 0, 2).to(device)
            else:
                max_trg_len = max(trg_lengths)
                batch_size = src_grid_seqs.size(0)
                constraint_mat = torch.zeros(max_trg_len, batch_size, parameters.id_size).to(device)
                pre_grids = torch.zeros(max_trg_len, batch_size, 3).to(device)
                next_grids = torch.zeros(max_trg_len, batch_size, 3).to(device)

            SE = SE.to(device)
            src_grid_seqs = src_grid_seqs.to(device)
            src_eid_seqs = src_eid_seqs.to(device)
            
            src_road_index_seqs = src_road_index_seqs.long().to(device)
            src_rate_seqs = src_rate_seqs.to(device)
            trg_in_grid_seqs = trg_in_grid_seqs.to(device)
            trg_rids = trg_rids.permute(1, 0, 2).long().to(device)
            trg_rates = trg_rates.permute(1, 0, 2).to(device)
            trg_in_index_seqs = trg_in_index_seqs.to(device)
            trg_gps_seqs = trg_gps_seqs.permute(1, 0, 2).to(device) # [trg len, batch size, 2]
            
            tra_time_A, tra_loca_A = build_graph(src_lengths, src_grid_seqs, src_gps_seqs)
            tra_time_A, tra_loca_A = tra_time_A.to(device), tra_loca_A.to(device)
            road_condition = torch.tensor(road_condition, dtype=torch.float).to(device)
            spatial_A_trans = torch.tensor(spatial_A_trans, dtype=torch.float).to(device)

            use_id_seq = torch.tensor(np.array(use_id_seq), dtype=torch.long).to(device)
            use_id_seq = use_id_seq.unsqueeze(0)
            
            output_ids, output_rates = model(use_id_seq, spatial_A_trans, road_condition, src_road_index_seqs, SE, tra_time_A, tra_loca_A, src_lengths, src_grid_seqs, src_eid_seqs, src_rate_seqs, trg_in_grid_seqs, 
                    trg_in_index_seqs, trg_rids, trg_rates, trg_lengths,pre_grids, next_grids, constraint_mat,
                                             src_pro_feas, online_features_dict, rid_features_dict,
                                             teacher_forcing_ratio=0)
            
            output_rates = output_rates.squeeze(2)
            output_seqs = toseq(rn_dict, output_ids, output_rates, parameters) # [trg len, batch size, 2]
            trg_rids = trg_rids.squeeze(2)
            trg_rates = trg_rates.squeeze(2)
            
            trg_lengths_sub = [length - 1 for length in trg_lengths]
            loss_ids1, recall, precision = cal_id_acc_batch(output_ids[1:], trg_rids[1:], trg_lengths_sub)
            dis_mae_loss, dis_rmse_loss, dis_rn_mae_loss, dis_rn_rmse_loss = cal_diss_loss_batch(output_seqs[1:],
                                                                                                 trg_gps_seqs[1:],
                                                                                                 output_ids[1:],
                                                                                                 trg_rids[1:],
                                                                                                 output_rates[1:],
                                                                                                 trg_rates[1:],
                                                                                                 trg_lengths_sub,
                                                                                                 True, rn, raw_rn_dict, new2raw_rid_dict)
            # for bbp
            output_ids_dim = output_ids.shape[-1]
            output_ids = output_ids[1:].reshape(-1,
                                                output_ids_dim)  # [(trg len - 1)* batch size, output id one hot dim]
            trg_rids = trg_rids[1:].reshape(-1)  # [(trg len - 1) * batch size],
            loss_ids = criterion_ce(output_ids, trg_rids)
            # rate loss
            loss_rates = criterion_reg(output_rates[1:], trg_rates[1:]) * parameters.lambda1
            
            # print("..................................................")
            epoch_dis_mae_loss += dis_mae_loss
            epoch_dis_rmse_loss += dis_rmse_loss
            epoch_dis_rn_mae_loss += dis_rn_mae_loss
            epoch_dis_rn_rmse_loss += dis_rn_rmse_loss
            epoch_id1_loss += loss_ids1
            epoch_recall_loss += recall
            epoch_precision_loss += precision
            epoch_rate_loss += loss_rates.item()
            epoch_id_loss += loss_ids.item()
            
        

        return epoch_id1_loss / len(iterator), epoch_recall_loss / len(iterator), \
               epoch_precision_loss / len(iterator), \
               epoch_dis_mae_loss / len(iterator), epoch_dis_rmse_loss / len(iterator), \
               epoch_dis_rn_mae_loss / len(iterator), epoch_dis_rn_rmse_loss / len(iterator), \
               epoch_rate_loss / len(iterator), epoch_id_loss / len(iterator)
    
def evaluate(model, spatial_A_trans, road_condition, SE, iterator, rn_dict, grid_rn_dict, rn, raw2new_rid_dict,
             online_features_dict, rid_features_dict, raw_rn_dict, new2raw_rid_dict, parameters):
    model.eval()  # must have this line since it will affect dropout and batch normalization
    
    epoch_dis_mae_loss = 0
    epoch_dis_rmse_loss = 0
    epoch_dis_rn_mae_loss = 0
    epoch_dis_rn_rmse_loss = 0
    epoch_id1_loss = 0
    epoch_recall_loss = 0
    epoch_precision_loss = 0
    epoch_rate_loss = 0
    epoch_id_loss = 0 # loss from dl model
    criterion_ce = nn.NLLLoss()
    criterion_reg = nn.MSELoss()

    with torch.no_grad():  # this line can help speed up evaluation
        for i, batch in enumerate(iterator):
            src_grid_seqs, src_gps_seqs, src_road_index_seqs, src_eid_seqs, src_rate_seqs, trg_in_t_seqs, trg_in_index_seqs, trg_in_grid_seqs, trg_in_gps_seqs, \
                    src_pro_feas, src_lengths, trg_gps_seqs, trg_rids, trg_rates, trg_lengths, use_id_seq = batch

            if parameters.dis_prob_mask_flag:
                constraint_mat, pre_grids, next_grids = get_constraint_mask(src_grid_seqs, src_gps_seqs, src_lengths,
                                                                            trg_lengths, grid_rn_dict, rn,
                                                                            raw2new_rid_dict, parameters)
                constraint_mat = constraint_mat.permute(1, 0, 2).to(device)
                pre_grids = pre_grids.permute(1, 0, 2).to(device)
                next_grids = next_grids.permute(1, 0, 2).to(device)
            else:
                max_trg_len = max(trg_lengths)
                batch_size = src_grid_seqs.size(0)
                constraint_mat = torch.zeros(max_trg_len, batch_size, parameters.id_size).to(device)
                pre_grids = torch.zeros(max_trg_len, batch_size, 3).to(device)
                next_grids = torch.zeros(max_trg_len, batch_size, 3).to(device)

            SE = SE.to(device)
            src_grid_seqs = src_grid_seqs.to(device)
            src_eid_seqs = src_eid_seqs.to(device)
            
            src_road_index_seqs = src_road_index_seqs.long().to(device)
            src_rate_seqs = src_rate_seqs.to(device)
            trg_in_grid_seqs = trg_in_grid_seqs.to(device)
            trg_rids = trg_rids.permute(1, 0, 2).long().to(device)
            trg_rates = trg_rates.permute(1, 0, 2).to(device)
            trg_in_index_seqs = trg_in_index_seqs.to(device)
            trg_gps_seqs = trg_gps_seqs.permute(1, 0, 2).to(device) # [trg len, batch size, 2]
            
            tra_time_A, tra_loca_A = build_graph(src_lengths, src_grid_seqs, src_gps_seqs)
            tra_time_A, tra_loca_A = tra_time_A.to(device), tra_loca_A.to(device)
            road_condition = torch.tensor(road_condition, dtype=torch.float).to(device)
            spatial_A_trans = torch.tensor(spatial_A_trans, dtype=torch.float).to(device)

            use_id_seq = torch.tensor(np.array(use_id_seq), dtype=torch.long).to(device)
            use_id_seq = use_id_seq.unsqueeze(0)
            output_ids, output_rates = model(use_id_seq, spatial_A_trans, road_condition, src_road_index_seqs, SE, tra_time_A, tra_loca_A, src_lengths, src_grid_seqs, src_eid_seqs, src_rate_seqs, trg_in_grid_seqs, 
                    trg_in_index_seqs, trg_rids, trg_rates, trg_lengths,pre_grids, next_grids, constraint_mat,
                                             src_pro_feas, online_features_dict, rid_features_dict,
                                             teacher_forcing_ratio=0)
            
            output_rates = output_rates.squeeze(2)
            output_seqs = toseq(rn_dict, output_ids, output_rates, parameters) # [trg len, batch size, 2]
            trg_rids = trg_rids.squeeze(2)
            trg_rates = trg_rates.squeeze(2)
            
            trg_lengths_sub = [length - 1 for length in trg_lengths]
            loss_ids1, recall, precision = cal_id_acc_batch(output_ids[1:], trg_rids[1:], trg_lengths_sub)
            dis_mae_loss, dis_rmse_loss, dis_rn_mae_loss, dis_rn_rmse_loss = cal_diss_loss_batch(output_seqs[1:],
                                                                                                 trg_gps_seqs[1:],
                                                                                                 output_ids[1:],
                                                                                                 trg_rids[1:],
                                                                                                 output_rates[1:],
                                                                                                 trg_rates[1:],
                                                                                                 trg_lengths_sub,
                                                                                                 False, rn, raw_rn_dict, new2raw_rid_dict)
            # for bbp
            output_ids_dim = output_ids.shape[-1]
            output_ids = output_ids[1:].reshape(-1,
                                                output_ids_dim)  # [(trg len - 1)* batch size, output id one hot dim]
            trg_rids = trg_rids[1:].reshape(-1)  # [(trg len - 1) * batch size],
            loss_ids = criterion_ce(output_ids, trg_rids)
            # rate loss
            loss_rates = criterion_reg(output_rates[1:], trg_rates[1:]) * parameters.lambda1
            
            # print("..................................................")
            epoch_dis_mae_loss += dis_mae_loss
            epoch_dis_rmse_loss += dis_rmse_loss
            
            epoch_dis_rn_mae_loss += dis_rn_mae_loss if dis_rn_mae_loss != None else 0
            epoch_dis_rn_rmse_loss += dis_rn_rmse_loss if dis_rn_rmse_loss != None else 0
            epoch_id1_loss += loss_ids1
            epoch_recall_loss += recall
            epoch_precision_loss += precision
            epoch_rate_loss += loss_rates.item()
            epoch_id_loss += loss_ids.item()
        
        return epoch_id1_loss / len(iterator), epoch_recall_loss / len(iterator), \
               epoch_precision_loss / len(iterator), \
               epoch_dis_mae_loss / len(iterator), epoch_dis_rmse_loss / len(iterator), \
               epoch_dis_rn_mae_loss / len(iterator), epoch_dis_rn_rmse_loss / len(iterator), \
               epoch_rate_loss / len(iterator), epoch_id_loss / len(iterator)
