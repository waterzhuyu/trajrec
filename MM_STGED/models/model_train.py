import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import toseq, get_constraint_mask

from models.model_utils import AttrDict

from models.loss_fn import cal_id_acc, check_rn_dis_loss

def train(model, iterator, optimizer, model_parameters):
    """
    Train the model with the given parameters
    """
    model.train() # Set the model to training mode
    
    criterion_reg = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss()
    
    epoch_ttl_loss = 0
    epoch_id1_loss = 0
    epoch_recall_loss = 0
    epoch_precision_loss = 0
    epoch_train_id_loss = 0
    epoch_rate_loss = 0
    
    grid_rn_dict = model_parameters.grid_rn_dict
    rn = model_parameters.rn
    raw2new_rid_dict = model_parameters.raw2new_rid_dict
    
    parameters = AttrDict()
    parameters.dis_prob_mask_flag = model_parameters.dis_prob_mask_flag
    parameters.device = model_parameters.device
    parameters.beta = model_parameters.beta
    parameters.id_size = model_parameters.id_size
    parameters.search_dist = model_parameters.search_dist
    
    for i, batch in enumerate(iterator):
        """
        generate the input and target sequences for training
        """
        src_grid_seqs, src_gps_seqs, src_eid_seqs, src_rate_seqs, src_road_index_seqs, src_pro_feas, \
        trg_gps_seqs, trg_eid_seqs, trg_rate_seqs, trg_t_seqs, trg_index_seqs, trg_interpolated_gps_seqs, trg_interpolated_grid_seqs, \
        src_lengths, trg_lengths = batch
        
        if parameters.dis_prob_mask_flag:
            constraint_mat, pre_grids, next_grids = get_constraint_mask(src_grid_seqs, src_gps_seqs, src_lengths,
                                                                        trg_lengths, grid_rn_dict, rn, raw2new_rid_dict, parameters)
            constraint_mat = constraint_mat.permute(1, 0, 2).to(parameters.device)
        else:
            max_trg_len = max(trg_lengths)
            batch_size = src_grid_seqs.size(0)
            constraint_mat = torch.zeros(max_trg_len, batch_size, parameters.id_size, device=parameters.device)
            
        src_pro_feas = src_pro_feas.float().to(parameters.device)
        
        src_grid_seqs = src_grid_seqs.permute(1, 0, 2).to(parameters.device)
        src_eid_seqs = src_eid_seqs.permute(1, 0, 2).to(parameters.device)
        src_rate_seqs = src_rate_seqs.permute(1, 0, 2).to(parameters.device)
        
        trg_gps_seqs = trg_gps_seqs.permute(1, 0, 2).to(parameters.device)
        trg_eid_seqs = trg_eid_seqs.permute(1, 0, 2).to(parameters.device)
        trg_rate_seqs = trg_rate_seqs.permute(1, 0, 2).to(parameters.device)
        
        """
        start training
        model: my model
        """
        optimizer.zero_grad()
        output_ids, output_rates = model(src_grid_seqs, src_lengths, trg_eid_seqs, trg_rate_seqs, trg_lengths, constraint_mat, teacher_forcing_ratio=0.5)
        
        output_rates = output_rates.squeeze(2)
        trg_rates = trg_rates.squeeze(2)
        
        trg_rids = trg_rids.squeeze(2)
        loss_ids1, recall, precision = cal_id_acc(output_ids[1:], trg_rids[1:], trg_lengths)
        
        
        """
        calculate for bbp
        """
        output_ids_dim = output_ids.shape[-1]
        output_ids = output_ids[1:].reshape(-1, output_ids_dim)  # [(trg len - 1)* batch size, output id one hot dim]
        trg_rids = trg_rids[1:].reshape(-1)  # [(trg len - 1) * batch size],
        
        loss_train_ids = criterion_ce(output_ids, trg_rids)
        loss_rates = criterion_reg(output_rates[1:], trg_rates[1:]) * parameters.lambda1
        ttl_loss = loss_train_ids + loss_rates
        
        ttl_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), parameters.clip)  # log_vars are not necessary to clip
        
        optimizer.step()
        
        
        epoch_ttl_loss += ttl_loss.item()
        epoch_id1_loss += loss_ids1
        epoch_recall_loss += recall
        epoch_precision_loss += precision
        epoch_train_id_loss += loss_train_ids.item()
        epoch_rate_loss += loss_rates.item()
    # exit()
    return epoch_ttl_loss / len(iterator), epoch_id1_loss / len(iterator), epoch_recall_loss / len(iterator), \
           epoch_precision_loss / len(iterator), epoch_rate_loss / len(iterator), epoch_train_id_loss / len(iterator)
        
            
    
    