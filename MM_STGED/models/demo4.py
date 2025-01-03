import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from copy import deepcopy
from einops import repeat

def mask_log_softmax(x, mask, log_flag=True):
    maxes = torch.max(x, 1, keepdim=True)[0]
    x_exp = torch.exp(x - maxes) * mask
    x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
    if log_flag:
        output_custom = torch.log(x_exp / x_exp_sum)
    else:
        output_custom = x_exp / x_exp_sum
    return output_custom

class Edge_merge(nn.Module):
    def __init__(self, node_in_channel, edge_in_channel, out_channel):
        super(Edge_merge, self).__init__()
        self.Z = nn.Linear(edge_in_channel, out_channel)
        self.H = nn.Linear(node_in_channel, out_channel)
    
    def forward(self, edge, node):
        # edge: B,T,T,2
        # node: B,T,F
        edge_transform = self.Z(edge)  # B,T,T,F

        node_transform = self.H(node)  # B,T,F
        node_transform_i = node_transform.unsqueeze(2) #B,1,T,F
        node_transform_j = node_transform.unsqueeze(1) #B,T,1,F
        return edge_transform + node_transform_i + node_transform_j  # B,T,T,F

class my_GCN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(my_GCN, self).__init__()
        self.linear1 = nn.Linear(in_channel, out_channel, bias=False)
        self.linear2 = nn.Linear(in_channel, out_channel, bias=False)
        
        self.wh = nn.Linear(out_channel, out_channel, bias=False)
        self.wtime = nn.Linear(out_channel, out_channel, bias=False)
        self.wloca = nn.Linear(out_channel, out_channel, bias=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.edge_merge = Edge_merge(in_channel, 2, out_channel)
        self.w_edge = nn.Linear(out_channel, out_channel, bias=False)

    def forward(self, X, A1, A2):
        # X: B,T,F
        # A: B,T,T
        A1X = torch.bmm(A1, X)
        AXW1 = self.relu(self.linear1(A1X))

        A2X = torch.bmm(A2, X)
        AXW2 = self.relu(self.linear2(A2X))

        A_merge = torch.cat((A1.unsqueeze(-1), A2.unsqueeze(-1)),dim=-1)  # B,T,T,2
        _edge_merge = self.edge_merge(A_merge, X)  #B,T,T,F
        _edge_merge = torch.sum(_edge_merge, dim=2) #B,T,F
        _merge = self.wh(X) + self.wtime(AXW1) + self.wloca(AXW2) + self.w_edge(_edge_merge)
        
        
        norm = self.bn(_merge.permute(0,2,1)).permute(0,2,1)
        all_state = X + self.relu(norm)
        hidden = torch.mean(all_state, 1).unsqueeze(0)
        return all_state.permute(1,0,2), hidden

class Encoder(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.hid_dim = parameters.hid_dim
        self.pro_output_dim = parameters.pro_output_dim
        self.online_features_flag = parameters.online_features_flag
        self.pro_features_flag = parameters.pro_features_flag

        input_dim = parameters.input_dim 
        # self.input_cat = nn.Linear(64 * 2, parameters.hid_dim, bias = False)
        # self.relu = nn.ReLU(inplace=True)
        # if self.online_features_flag:
            # input_dim = input_dim + parameters.online_dim

        self.rnn = nn.GRU(input_dim, self.hid_dim)
        # self.dropout = nn.Dropout(parameters.dropout)

        # if self.pro_features_flag:
        #     self.extra = Extra_MLP(parameters)
        #     self.fc_hid = nn.Linear(self.hid_dim + self.pro_output_dim, self.hid_dim)

    def forward(self, src, src_len, pro_features):
        packed_embedded = nn.utils.rnn.pack_padded_sequence(src, src_len)
        packed_outputs, hidden = self.rnn(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
            
        # if self.pro_features_flag:
        #     extra_emb = self.extra(pro_features)
        #     extra_emb = extra_emb.unsqueeze(0)
        #     # extra_emb = [1, batch size, extra output dim]
        #     hidden = torch.tanh(self.fc_hid(torch.cat((extra_emb, hidden), dim=2)))
        #     # hidden = [1, batch size, hid dim]

        return outputs, hidden



class SpecifiedEncoder(nn.Module):
    def __init__(self, hid_dim, pro_output_dim, input_dim, dropout):
        """config using some arguments but not a ArgDict"""
        super().__init__()
        self.hid_dim = hid_dim
        self.pro_output_dim = pro_output_dim

        input_dim = input_dim 

        self.rnn = nn.GRU(input_dim, self.hid_dim)


    def forward(self, src, src_len, pro_features):
        packed_embedded = nn.utils.rnn.pack_padded_sequence(src, src_len)
        packed_outputs, hidden = self.rnn(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
            
        return outputs, hidden  # (seq_len, batch_size, hid_dim), (num_layers * num_directions, batch_size, hid_dim)


class AttentionLayer(nn.Module):
    """
    Multi-head scaled dot attention
    """
    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(-1, -2)  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class ProjectedAttentionLayer(nn.Module):
    """
    Temporal projected attention layer
    """
    def __init__(self, dim_proj, d_model, n_heads, d_ff=None, dropout=0.1):
        super(ProjectedAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.out_attn = AttentionLayer(d_model, n_heads, mask=None)
        self.in_attn = AttentionLayer(d_model, n_heads, mask=None)
        self.projector = nn.Parameter(torch.randn(dim_proj, d_model))

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.MLP = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                 nn.Linear(d_ff, d_model))

        # self.seq_len = seq_len

    def forward(self, x):
        # x: [b s n d]
        batch = x.shape[0]
        # projector = repeat(self.projector, 'dim_proj d_model -> repeat seq_len dim_proj d_model',
        #                       repeat=batch, seq_len=self.seq_len)  # [b, s, c, d]
        projector = repeat(self.projector, 'dim_proj d_model -> repeat dim_proj d_model', repeat=batch)

        message_out = self.out_attn(projector, x, x)  # [b, s, c, d] <-> [b s n d] -> [b s c d]
        message_in = self.in_attn(x, projector, message_out)  # [b s n d] <-> [b, s, c, d] -> [b s n d]
        message = x + self.dropout(message_in)
        message = self.norm1(message)
        message = message + self.dropout(self.MLP(message))
        message = self.norm2(message)

        return message


class attention(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(attention, self).__init__()
        self.l_k = nn.Linear(in_channel, out_channel)
        self.l_q = nn.Linear(in_channel, out_channel)
        self.l_v = nn.Linear(in_channel, out_channel)
    def forward(self, x_k, x_q, mask=None, dropout=None):
        key = self.l_k(x_k)
        query = self.l_q(x_q)
        value = self.l_v(x_k)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # scores: (batch, N, h, T1, T2)

        if mask is not None:
            scores = scores.masked_fill_(mask == 0, -1e9)  # -1e9 means attention scores=0
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        # p_attn: (batch, N, h, T1, T2)

        return torch.matmul(p_attn, value)  # (batch, N, h, T1, d_k), (batch, N, h, T1, T2)



class Attention(nn.Module):
    # TODO update to more advanced attention layer.
    def __init__(self, parameters):
        super().__init__()
        self.hid_dim = parameters.hid_dim

        self.attn = nn.Linear(self.hid_dim * 2, self.hid_dim)
        self.v = nn.Linear(self.hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, attn_mask):
        # hidden = [1, bath size, hid dim]
        # encoder_outputs = [src len, batch size, hid dim * num directions]
        src_len = encoder_outputs.shape[0]
        # repeat decoder hidden sate src_len times
        hidden = hidden.repeat(src_len, 1, 1)
        hidden = hidden.permute(1, 0, 2)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # hidden = [batch size, src len, hid dim]
        # encoder_outputs = [batch size, src len, hid dim * num directions]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch size, src len, hid dim]

        attention = self.v(energy).squeeze(2)
        # attention = [batch size, src len]
        attention = attention.masked_fill(attn_mask == 0, -1e10)
        # using mask to force the attention to only be over non-padding elements.

        return F.softmax(attention, dim=1)


class DecoderMulti(nn.Module):
    def __init__(self, parameters):
        super().__init__()

        self.id_size = parameters.id_size
        self.id_emb_dim = parameters.id_emb_dim
        self.hid_dim = parameters.hid_dim
        self.pro_output_dim = parameters.pro_output_dim
        self.online_dim = parameters.online_dim
        self.rid_fea_dim = parameters.rid_fea_dim

        self.attn_flag = parameters.attn_flag
        self.dis_prob_mask_flag = parameters.dis_prob_mask_flag  # final softmax
        self.online_features_flag = parameters.online_features_flag
        self.tandem_fea_flag = parameters.tandem_fea_flag

        self.top_k = parameters.top_K
        rnn_input_dim = self.id_emb_dim + 1 + 64
        fc_id_out_input_dim = self.hid_dim
        fc_rate_out_input_dim = self.hid_dim
        
        type_input_dim = self.id_emb_dim + self.hid_dim
        self.tandem_fc = nn.Sequential(
                          nn.Linear(type_input_dim, self.hid_dim),
                          nn.ReLU()
                          )
        self.user_merge_layer = nn.Sequential(
            nn.Linear(self.hid_dim + 256, self.hid_dim)  #TODO: change this line!
        )
        if self.attn_flag:
            self.attn = Attention(parameters)
            rnn_input_dim = rnn_input_dim + self.hid_dim 

        if self.online_features_flag:
            rnn_input_dim = rnn_input_dim + self.online_dim  # 5 poi and 5 road network
            
        if self.tandem_fea_flag:
            fc_rate_out_input_dim = self.hid_dim + self.rid_fea_dim
        
        self.rnn = nn.GRU(rnn_input_dim, self.hid_dim)
        self.fc_rate_out = nn.Linear(fc_rate_out_input_dim, 1)
        self.dropout = nn.Dropout(parameters.dropout)

    def forward2id(self, emb_id, decoder_node2vec, road_index, spatial_A_trans, topk_mask, trg_index, input_id, input_rate, hidden, encoder_outputs, attn_mask,
                pre_grid, next_grid, constraint_vec, pro_features, online_features, rid_features):

        input_id = input_id.squeeze(1).unsqueeze(0)  # cannot use squeeze() bug for batch size = 1
        input_rate = input_rate.unsqueeze(0)
        embedded = self.dropout(emb_id(input_id))

        if self.attn_flag:
            a = self.attn(hidden, encoder_outputs, attn_mask)
            a = a.unsqueeze(1)
            encoder_outputs = encoder_outputs.permute(1, 0, 2)
            weighted = torch.bmm(a, encoder_outputs)
            weighted = weighted.permute(1, 0, 2)

            if self.online_features_flag:
                rnn_input = torch.cat((weighted, embedded, input_rate, 
                                       online_features.unsqueeze(0)), dim=2)
            else:
                rnn_input = torch.cat((weighted, embedded, input_rate), dim=2)
        else:
            if self.online_features_flag:
                rnn_input = torch.cat((embedded, input_rate, online_features.unsqueeze(0)), dim=2)
            else:
                rnn_input = torch.cat((embedded, input_rate), dim=2)
        rnn_input = torch.cat((rnn_input, road_index.unsqueeze(0)), dim=2)
        
        
        output, hidden = self.rnn(rnn_input, hidden)
        
        tra_vec = torch.mean(decoder_node2vec, dim=1).unsqueeze(0)
        user_merge = self.user_merge_layer(torch.cat((output, tra_vec), dim=2))

        if self.dis_prob_mask_flag:
            if topk_mask is not None:
                trg_index_repeat = trg_index.repeat(1, constraint_vec.shape[1])
                _tmp_mask = 0
                for i in range(self.top_k):
                    id_index = topk_mask[:,i:i+1].squeeze(1).long()
                    _tmp_mask = _tmp_mask + spatial_A_trans[id_index]
                _tmp_mask[_tmp_mask>1] = 1.
                constraint_vec = torch.where(trg_index_repeat==1, constraint_vec, _tmp_mask) 
        return user_merge, constraint_vec, hidden
    
    def forward2rate(self, emb_id, prediction_id, user_merge, rid_features):
        # pre_rate
        max_id = prediction_id.argmax(dim=1).long()
        id_emb = self.dropout(emb_id(max_id))
        rate_input = torch.cat((id_emb, user_merge.squeeze(0)),dim=1)
        rate_input = self.tandem_fc(rate_input)  # [batch size, hid dim]
        if self.tandem_fea_flag:
            prediction_rate = torch.sigmoid(self.fc_rate_out(torch.cat((rate_input, rid_features), dim=1)))
        else:
            prediction_rate = torch.sigmoid(self.fc_rate_out(rate_input))

        return prediction_rate
        
    def forward(self, decoder_node2vec, user_id, road_index, spatial_A_trans, topk_mask, trg_index, input_id, input_rate, hidden, encoder_outputs, attn_mask,
                pre_grid, next_grid, constraint_vec, pro_features, online_features, rid_features):

        input_id = input_id.squeeze(1).unsqueeze(0)  # cannot use squeeze() bug for batch size = 1
        input_rate = input_rate.unsqueeze(0)
        embedded = self.dropout(self.emb_id(input_id))

        if self.attn_flag:
            a = self.attn(hidden, encoder_outputs, attn_mask)
            a = a.unsqueeze(1)
            encoder_outputs = encoder_outputs.permute(1, 0, 2)
            weighted = torch.bmm(a, encoder_outputs)
            weighted = weighted.permute(1, 0, 2)

            if self.online_features_flag:
                rnn_input = torch.cat((weighted, embedded, input_rate, 
                                       online_features.unsqueeze(0)), dim=2)
            else:
                rnn_input = torch.cat((weighted, embedded, input_rate), dim=2)
        else:
            if self.online_features_flag:
                rnn_input = torch.cat((embedded, input_rate, online_features.unsqueeze(0)), dim=2)
            else:
                rnn_input = torch.cat((embedded, input_rate), dim=2)
        rnn_input = torch.cat((rnn_input, road_index.unsqueeze(0)), dim=2)
        
        
        output, hidden = self.rnn(rnn_input, hidden)
        
        user_info = self.user_embedding(user_id)
        tra_vec = torch.mean(decoder_node2vec, dim=1).unsqueeze(0)
        user_merge = self.user_merge_layer(torch.cat((output, user_info, tra_vec), dim=2))

        if self.dis_prob_mask_flag:
            if topk_mask is not None:
                trg_index_repeat = trg_index.repeat(1, constraint_vec.shape[1])
                _tmp_mask = 0
                for i in range(self.top_k):
                    id_index = topk_mask[:,i:i+1].squeeze(1).long()
                    _tmp_mask = _tmp_mask + spatial_A_trans[id_index]
                _tmp_mask[_tmp_mask>1] = 1.
                constraint_vec = torch.where(trg_index_repeat==1, constraint_vec, _tmp_mask)
            prediction_id = mask_log_softmax(self.fc_id_out(user_merge.squeeze(0)), 
                                             constraint_vec, log_flag=True)
        else:
            prediction_id = F.log_softmax(self.fc_id_out(user_merge.squeeze(0)), dim=1)

        # pre_rate
        max_id = prediction_id.argmax(dim=1).long()
        id_emb = self.dropout(self.emb_id(max_id))
        rate_input = torch.cat((id_emb, user_merge.squeeze(0)),dim=1)
        rate_input = self.tandem_fc(rate_input)  # [batch size, hid dim]
        if self.tandem_fea_flag:
            prediction_rate = torch.sigmoid(self.fc_rate_out(torch.cat((rate_input, rid_features), dim=1)))
        else:
            prediction_rate = torch.sigmoid(self.fc_rate_out(rate_input))

        return prediction_id, prediction_rate, hidden


class spatialTemporalConv(nn.Module):
    def __init__(self, in_channel, base_channel):
        super(spatialTemporalConv, self).__init__()
        self.start_conv = nn.Conv2d(in_channel, base_channel, 1, 1, 0)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(base_channel, base_channel, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base_channel)
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(base_channel, base_channel, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
    def forward(self, road_condition):
        #road_condition: T, N, N
        T, N, N = road_condition.shape
        # print(road_condition.unsqueeze(-1).shape)
        _start = self.start_conv(road_condition.unsqueeze(1))  # T,1,N,N
        spatialConv = self.spatial_conv(_start)  #T,F,N,N
        spatial_reshape = spatialConv.reshape(T, -1, N*N).permute(2, 1, 0) # N*N,F,T
        temporalConv = self.temporal_conv(spatial_reshape)
        conv_res = temporalConv.reshape(N, N, -1, T).permute(3, 2, 0, 1)  # T,F,N,N
        # print((_start + conv_res).shape)
        return (_start + conv_res).permute(0, 2, 3, 1)


class SharedEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.shared_embedding = nn.init.xavier_uniform_(nn.Parameter(torch.empty(args.num_emb, args.emb_dim)))


class DecoderHead(nn.Module):
    """V1: shared rate head"""
    def __init__(self, hid_dim, id_size, dis_prob_mask_flag):
        super().__init__()
        self.dis_prob_mask_flag = dis_prob_mask_flag
        self.fc_id_out = nn.Linear(hid_dim, id_size)
        
    def forward(self, user_merge, constraint_vec):
        if self.dis_prob_mask_flag:
            prediction_id = mask_log_softmax(self.fc_id_out(user_merge.squeeze(0)), constraint_vec, log_flag=True)
        else:
            prediction_id = F.log_softmax(self.fc_id_out(user_merge.squeeze(0)), dim=1)

        return prediction_id


class MM_STGED(nn.Module):
    def __init__(self, encoder, decoder, shared_emb, num_emb, id_size, hid_dim, id_emb_dim, base_channel, num_layers, x_id, y_id, top_k):
        super(MM_STGED, self).__init__()
        self.encoder = encoder  # shared encoder
        #TODO: how to choose dropout ratio?
        self.specified_encoder = SpecifiedEncoder(hid_dim=512, pro_output_dim=8, input_dim=3, dropout=0.5)
        self.attn_layers_t = nn.ModuleList(
            [ProjectedAttentionLayer(dim_proj=10, d_model=hid_dim, n_heads=4, d_ff=1 * hid_dim, dropout=0.3) for _ in range(num_layers)]
        )
        self.shared_emb = shared_emb
        self.mlp = nn.Linear(num_emb, id_size)
        self.emb_id = nn.Embedding(id_size, id_emb_dim) 
        self.decoder = decoder
        self.decoder_head = DecoderHead(hid_dim, id_size, True)
        self.spatialTemporalConv = spatialTemporalConv(1, 64)
        self.id_size = id_size
        
        self.dropout = nn.Dropout(p=0.3)
        self.x_id = x_id
        self.y_id = y_id
        
        self.topK = top_k
        
        self.encoder_out = nn.Sequential(
            nn.Linear(512+256, 512), #TODO: change this line!
            nn.ReLU(inplace=True),
            nn.Linear(512, 512)
        )
        self.encoder_point_cat = nn.Sequential(
            nn.Linear(512+256, 512), #TODO: change this line!
            nn.ReLU(inplace=True),
            nn.Linear(512, 512)
        )
        self.mygcn = my_GCN(base_channel, base_channel)
        #TODO: for debugging
        self.device = "cuda:5"
        # self.device = "cpu"
        

    def forward(self, user_tf_idf, spatial_A_trans, road_condition, src_road_index_seqs, SE, 
                tra_time_A, tra_loca_A, src_len, src_grid_seqs, src_eid_seqs, src_rate_seqs, trg_in_grid_seqs,
                trg_in_index_seqs, trg_rids, trg_rates, trg_len,
                pre_grids, next_grids, constraint_mat, pro_features, 
                online_features_dict, rid_features_dict,
                teacher_forcing_ratio=0.5):
                
        batchsize, max_src_len, _ = src_grid_seqs.shape
        
        """Graph-based trajectory encoder"""
        src_attention, src_hidden = self.encoder(src_grid_seqs.permute(1,0,2), src_len, pro_features)
        src_attention = src_attention.permute(1, 0, 2)   # B, T, F

        """Src feed into specified encoder"""
        src_specified_attention, src_specified_hidden = self.specified_encoder(src_grid_seqs.permute(1, 0, 2), src_len, pro_features)
        src_specified_attention = src_specified_attention.permute(1, 0, 2)

        #FIXME: concat outputs of shared and specified encoder
        # src_attention = torch.cat((src_attention, src_specified_attention), dim=-1)
        # src_hidden = torch.cat((src_hidden, src_specified_hidden), dim=-1)
        src_attention = src_attention + src_specified_attention
        src_hidden = src_hidden + src_specified_hidden
        
        #FIXME: add ProjectedAttentionLayer
        for attn_t in self.attn_layers_t:
            src_attention = attn_t(src_attention)

        src_attention, src_hidden = self.mygcn(src_attention, tra_time_A, tra_loca_A)

        """add road"""
        #TODO: concat or add, which is better?
        shared_emb = self.mlp(self.shared_emb.shared_embedding.transpose(0, 1))
        shared_emb = shared_emb.transpose(0, 1)
        SE = torch.concat((SE, shared_emb), dim=-1)

        all_road_embed =  torch.einsum('btr,rd->btd',(constraint_mat.permute(1, 0, 2), SE))   # B, T, F
        summ = constraint_mat.permute(1, 0, 2).sum(-1).unsqueeze(-1)
        trajectory_point_embed = all_road_embed / summ   #  得到了每个节点的表示

        trajectory_point_sum = trg_in_index_seqs.sum(1)
        trajectory_embed = (trajectory_point_embed.sum(1) / trajectory_point_sum).unsqueeze(0)  # 得到每个轨迹的表示
        src_hidden = self.encoder_out(torch.cat((src_hidden, trajectory_embed), -1))   # 轨迹最终表示：路段表示+原始图表示

        # 接下来基于节点的表示trajectory_point_embed，将其拼接到GRU输出上
        _trg_in_index_seqs = trg_in_index_seqs.repeat(1, 1, 64)
        _imput_zero = torch.zeros((batchsize, 64)).to(self.device)
        trajectory_point_road = []

        trajectory_point_road = torch.zeros((max_src_len, batchsize, 256)).to(self.device)  #TODO: change this line!

        for batch in range(batchsize):
            # print(src_grid_seqs[0])
            _traject_i_index = trg_in_index_seqs[batch,:, 0]
            # print(_traject_i_index, _traject_i_index.sum())
            b = trajectory_point_embed[batch][_traject_i_index==1.]
            curr_traject_i_length = b.shape[0]
            # print(b.shape, '....', trajectory_point_road[1:1+curr_traject_i_length, batch].shape, '....', trajectory_point_road.shape)
            trajectory_point_road[1:1+curr_traject_i_length, batch] = b
            
        src_attention = self.encoder_point_cat(torch.cat((src_attention, trajectory_point_road), -1))

        # contextual road condition representation
        road_conv = self.spatialTemporalConv(road_condition)  # T, N, N, F

        #trajectory-related road condition
        road_index = None
        for i in range(1, max_src_len):  #ignore the first point
            TNN_i = src_road_index_seqs[:,i]  # B,3
            _tmp = road_conv[TNN_i[:,0], TNN_i[:,1], TNN_i[:,2]].unsqueeze(1)    # B,T, F
            if i == 1:
                road_index = _tmp
            else:
                road_index = torch.cat((road_index, _tmp), 1)
        road_index = road_index.mean(1)
        
        
        max_trg_len = trg_rids.size(0)
        if self.decoder.attn_flag:
            attn_mask = torch.zeros(batchsize, max(src_len))  # only attend on unpadded sequence
            for i in range(len(src_len)):
                attn_mask[i][:src_len[i]] = 1.
            attn_mask = attn_mask.to(self.device)
        else:
            attn_mask = None

        outputs_id, outputs_rate = self.normal_step(SE, user_tf_idf, road_index, spatial_A_trans, trg_in_index_seqs, 
                                                    max_trg_len, batchsize, trg_rids, trg_rates, trg_len,
                                                    src_attention, src_hidden, attn_mask,
                                                    online_features_dict,
                                                    rid_features_dict,
                                                    pre_grids, next_grids, constraint_mat, pro_features,
                                                    teacher_forcing_ratio)

        return outputs_id, outputs_rate

    def normal_step(self, SE, user_tf_idf, road_index, spatial_A_trans, trg_in_index_seqs, max_trg_len, batch_size, trg_id, trg_rate, trg_len, encoder_outputs, hidden,
                    attn_mask, online_features_dict, rid_features_dict,
                    pre_grids, next_grids, constraint_mat, pro_features, teacher_forcing_ratio):
        """
        Returns:
        -------
        outputs_id: [seq len, batch size, id size]
        outputs_rate: [seq len, batch size, 1]
        """
        # tensor to store decoder outputs
        outputs_id = torch.zeros(max_trg_len, batch_size, self.id_size).to(self.device)
        outputs_rate = torch.zeros(trg_rate.size()).to(self.device)

        # first input to the decoder is the <sos> tokens
        input_id = trg_id[0, :]
        input_rate = trg_rate[0, :]
        
        decoder_node2vec = SE[input_id.long()]  # index from SE
        topk_mask = None
        for t in range(1, max_trg_len):
            trg_index = trg_in_index_seqs[:,t]   #batchsize 
            if self.decoder.online_features_flag:
                online_features = get_dict_info_batch(input_id, online_features_dict).to(self.device)
            else:
                online_features = torch.zeros((1, batch_size, self.decoder.online_dim))
            if self.decoder.tandem_fea_flag:
                rid_features = get_dict_info_batch(input_id, rid_features_dict).to(self.device)
            else:
                rid_features = None
            # prediction_id, prediction_rate, hidden = self.decoder(decoder_node2vec, user_tf_idf, road_index, spatial_A_trans, topk_mask, trg_index, input_id, input_rate, hidden, encoder_outputs,
            #                                                          attn_mask, pre_grids[t], next_grids[t],
            #                                                          constraint_mat[t], pro_features, online_features,
            #                                                          rid_features)
            user_merge, constraint_vec, hidden = self.decoder.forward2id(self.emb_id, decoder_node2vec, road_index, spatial_A_trans, topk_mask, trg_index, input_id, input_rate, hidden, encoder_outputs,
                                                                     attn_mask, pre_grids[t], next_grids[t],
                                                                     constraint_mat[t], pro_features, online_features,
                                                                     rid_features)
            prediction_id = self.decoder_head(user_merge, constraint_vec)

            prediction_rate = self.decoder.forward2rate(self.emb_id, prediction_id, user_merge, None)

            # place predictions in a tensor holding predictions for each token
            outputs_id[t] = prediction_id
            outputs_rate[t] = prediction_rate

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1_id = prediction_id.argmax(1)
            top1_id = top1_id.unsqueeze(-1)  # make sure the output has the same dimension as input

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input_id = trg_id[t] if teacher_force else top1_id
            input_rate = trg_rate[t] if teacher_force else prediction_rate

            #topK mask
            topk_mask = prediction_id.topk(self.topK, dim=-1, sorted=True)[1]
            
            decoder_node2vec = torch.cat((decoder_node2vec, SE[input_id.long()]),dim=1)

        # max_trg_len, batch_size, trg_rid_size
        outputs_id = outputs_id.permute(1, 0, 2)  # batch size, seq len, rid size
        outputs_rate = outputs_rate.permute(1, 0, 2)  # batch size, seq len, 1
        for i in range(batch_size):
            outputs_id[i][trg_len[i]:] = 0
            outputs_id[i][trg_len[i]:, 0] = 1  # make sure argmax will return eid0
            outputs_rate[i][trg_len[i]:] = 0
        outputs_id = outputs_id.permute(1, 0, 2)
        outputs_rate = outputs_rate.permute(1, 0, 2)
        return outputs_id, outputs_rate
