import torch
import torch.nn as nn
import torch.nn.functional as F

import random

def mask_log_softmax(x, mask, log_flag=True):
    maxes = torch.max(x, 1, keepdim=True)[0]
    x_exp = torch.exp(x - maxes) * mask
    x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
    if log_flag:
        output_custom = torch.log(x_exp / x_exp_sum)
    else:
        output_custom = x_exp / x_exp_sum
    return output_custom

class Encoder(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.parameters = parameters
        self.hid_dim = parameters.hid_dim
    
        input_dim = 3
        
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=self.hid_dim)
        self.dropout = nn.Dropout(p=parameters.dropout)
        
    def forward(self, src, src_len):
        # src: (seq_len, batch_size, input_dim)
        # src_len: (batch_size)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(src, src_len)
        packed_outputs, hidden = self.rnn(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        # outputs: (seq_len, batch_size, hid_dim)
        # hidden: (1, batch_size, hid_dim)
        outputs = self.dropout(outputs)
        return outputs, hidden
    
class Attention(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.parameters = parameters
        self.hid_dim = parameters.hid_dim
        
        self.attn = nn.Linear(self.hid_dim * 2, self.hid_dim)
        self.v = nn.Linear(self.hid_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs, attn_mask):
        # hidden: (1, batch_size, hid_dim)
        # encoder_outputs: (seq_len, batch_size, hid_dim * num directions(2))
        # attn_mask: (batch_size, seq_len)
        
        src_len = encoder_outputs.shape[0]
        
        hidden = hidden.repeat(src_len, 1, 1)
        hidden = hidden.permute(1, 0, 2)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        attention = attention.masked_fill(attn_mask == 0, -1e10)
        attention = F.softmax(attention, dim=1)
        # attention: (batch_size, seq_len)
        
        return attention
    
class DecoderMulti(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.parameters = parameters
        self.id_size = parameters.id_size
        self.id_emb_dim = parameters.id_emb_dim
        self.hid_dim = parameters.hid_dim
        
        self.attn_flag = parameters.attn_flag
        self.dis_prob_mask_flag = parameters.dis_prob_mask_flag
        
        self.emd_id = nn.Embedding(self.id_size, self.id_emb_dim)
        
        rnn_input_dim = self.id_emb_dim + 1
        fc_id_out_input_dim = self.hid_dim
        fc_rate_out_input_dim = self.hid_dim 
        
        if self.attn_flag:
            self.attn = Attention(parameters)
            rnn_input_dim += self.hid_dim
            
        
        self.rnn = nn.GRU(input_size=rnn_input_dim, hidden_size=self.hid_dim)
        
        self.fc_id_out = nn.Linear(fc_id_out_input_dim, self.id_size)
        self.fc_rate_out = nn.Linear(fc_rate_out_input_dim, 1)
        
        self.dropout = nn.Dropout(p=parameters.dropout)
        
    def forward(self, input_id, input_rate, hidden, encoder_outputs, attn_mask, constraint_vec):
        # input_id: (batch_size, 1)
        # input_rate: (batch_size, 1)
        # hidden: (1, batch_size, hid_dim)
        # encoder_outputs: (seq_len, batch_size, hid_dim * num directions(2))
        # attn_mask: (batch_size, seq_len)
        # constraint_vec: (batch_size, id_size), id_size is the vector of reachable rid
        
        input_id = input_id.squeeze(1).unsqueeze(0)
        input_rate = input_rate.unsqueeze(0)
        embedded = self.dropout(self.emd_id(input_id))
        
        if self.attn_flag:
            a = self.attn(hidden, encoder_outputs, attn_mask)
            # a = [batch size, src len]
            a = a.unsqueeze(1)
            # a = [batch size, 1, src len]
            encoder_outputs = encoder_outputs.permute(1, 0, 2)
            # encoder_outputs = [batch size, src len, hid dim * num directions]
            weighted = torch.bmm(a, encoder_outputs)
            # weighted = [batch size, 1, hid dim * num directions]
            weighted = weighted.permute(1, 0, 2)
            # weighted = [1, batch size, hid dim * num directions]

            rnn_input = torch.cat((weighted, embedded, input_rate), dim=2)
            
        else:
            rnn_input = torch.cat((embedded, input_rate), dim=2)
            
        output, hidden = self.rnn(rnn_input, hidden)
        # output: (1, batch_size, hid_dim)
        # hidden: (1, batch_size, hid_dim)
        assert (output == hidden).all()
        
        # pre_rid
        if self.dis_prob_mask_flag:
            prediction_id = mask_log_softmax(self.fc_id_out(output.squeeze(0)), 
                                             constraint_vec, log_flag=True)
        else:
            prediction_id = F.log_softmax(self.fc_id_out(output.squeeze(0)), dim=1)
            # then the loss function should change to nll_loss()
            
        # pre_rate
        max_id = prediction_id.argmax(dim=1).long()
        id_emb = self.dropout(self.emb_id(max_id))
        rate_input = torch.cat((id_emb, hidden.squeeze(0)),dim=1)
        prediction_rate = torch.sigmoid(self.fc_rate_out(rate_input))
        
        # predication_id: (batch_size, id_size)
        # prediction_rate: (batch_size, 1)
        
        return prediction_id, prediction_rate, hidden
    
class Seq2SeqMulti(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, src_len, trg_id, trg_rate, trg_len, constraint_mat, teacher_forcing_ratio=0.5):
        """
        src: (seq_len, batch_size, input_dim(3)) (x, y, t)
        src_len: (batch_size)
        trg_id: (trg_len, batch_size, 1)
        trg_rate: (trg_len, batch_size, 1)
        trg_len: (batch_size)
        pre_grids: (trg_len, batch_size, 3)
        next_grids: (trg_len, batch_size, 3)
        constraint_mat: (trg_len, batch_size, id_size)
        teacher_forcing_ratio: the probability of using teacher forcing
        e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        ------
        Returns:
        outpid_id: (trg_len, batch_size, id_size(1)) 
        out_rate: (trg_len, batch_size, 1)
        """        
        
        max_trg_len = trg_id.size(0)
        batch_size = trg_id.size(1)
        
        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hiddens = self.encoder(src, src_len)
        
        if self.decoder.attn_flag:
            attn_mask = torch.zeros(batch_size, max(src_len))  # only attend on unpadded sequence
            for i in range(len(src_len)):
                attn_mask[i][:src_len[i]] = 1.
            attn_mask = attn_mask.to(self.device)
        else:
            attn_mask = None
            
        outputs_id, outputs_rate = self.normal_step(max_trg_len, batch_size, trg_id, trg_rate, trg_len,
                                                encoder_outputs, hiddens, attn_mask, constraint_mat, teacher_forcing_ratio)
        
        return outputs_id, outputs_rate
    
    def normal_step(self, max_trg_len, batch_size, trg_id, trg_rate, trg_len,
                    encoder_outputs, hiddens, attn_mask, constraint_mat, teacher_forcing_ratio):
        """
        Returns:
        outputs_id: (trg_len, batch_size, id_size(1))
        outputs_rate: (trg_len, batch_size, 1)
        """
        # tensor to store decoder outputs
        outputs_id = torch.zeros(max_trg_len, batch_size, self.decoder.id_size).to(self.device)
        outputs_rate = torch.zeros(trg_rate.size()).to(self.device)
        
        # first input to the decoder is the <sos> tokens
        input_id = trg_id[0, :]
        input_rate = trg_rate[0, :]
        
        for t in range(1, max_trg_len):
            # insert input token embedding, previous hidden state, all encoder hidden states and attn_mask
            # receive output tensor (predictions) and new hidden state
            prediction_id, prediction_rate, hidden = self.decoder(input_id, input_rate, hidden, encoder_outputs,
                                                                    attn_mask, constraint_mat[t])

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