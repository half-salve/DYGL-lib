import logging

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

'''
@article{DBLP:journals/corr/abs-2002-07962,
  author    = {Da Xu and
               Chuanwei Ruan and
               Evren K{\"{o}}rpeoglu and
               Sushant Kumar and
               Kannan Achan},
  title     = {Inductive Representation Learning on Temporal Graphs},
  journal   = {CoRR},
  volume    = {abs/2002.07962},
  year      = {2020},
  url       = {https://arxiv.org/abs/2002.07962},
  eprinttype = {arXiv},
  eprint    = {2002.07962},
  timestamp = {Mon, 02 Mar 2020 16:46:06 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2002-07962.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}


'''
from .baseEncode import TimeEncode,PosEncode,EmptyEncode

class MergeLayer(nn.Module): #torch.nn.Bilinear(self.feat_dim, self.feat_dim, 1, bias=True)
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        #self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

        # special linear layer for motif explainability

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        #x = self.layer_norm(x)
        h = self.act(self.fc1(x))
        z = self.fc2(h)
        return z
    

class LSTMPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim, time_dim):
        super(LSTMPool, self).__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.edge_dim = edge_dim
        
        self.att_dim = feat_dim + edge_dim + time_dim
        
        self.act = torch.nn.ReLU()
        
        self.lstm = torch.nn.LSTM(input_size=self.att_dim, 
                                  hidden_size=self.feat_dim, 
                                  num_layers=1, 
                                  batch_first=True)
        self.merger = MergeLayer(feat_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        # seq [B, N, D]
        # mask [B, N]
        seq_x = torch.cat([seq, seq_e, seq_t], dim=2)
            
        _, (hn, _) = self.lstm(seq_x)
        
        hn = hn[-1, :, :] #hn.squeeze(dim=0)

        out = self.merger.forward(hn, src)
        return out, None
    

class MeanPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim):
        super(MeanPool, self).__init__()
        self.edge_dim = edge_dim
        self.feat_dim = feat_dim
        self.act = torch.nn.ReLU()
        self.merger = MergeLayer(edge_dim + feat_dim, feat_dim, feat_dim, feat_dim)
        
    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        # seq [B, N, D]
        # mask [B, N]
        src_x = src
        seq_x = torch.cat([seq, seq_e], dim=2) #[B, N, De + D]
        hn = seq_x.mean(dim=1) #[B, De + D]
        output = self.merger(hn, src_x)
        return output, None

class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]
        attn = self.dropout(attn) # [n * b, l_v, d]
                
        output = torch.bmm(attn, v)
        
        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        #output = self.layer_norm(output)

        return output, attn
    

class MapBasedMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.wq_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wk_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wv_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.weight_map = nn.Linear(2 * d_k, 1, bias=False)
        
        nn.init.xavier_normal_(self.fc.weight)
        
        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=2)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.wq_node_transform(q).view(sz_b, len_q, n_head, d_k)
        
        k = self.wk_node_transform(k).view(sz_b, len_k, n_head, d_k)
        
        v = self.wv_node_transform(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        q = torch.unsqueeze(q, dim=2) # [(n*b), lq, 1, dk]
        q = q.expand(q.shape[0], q.shape[1], len_k, q.shape[3]) # [(n*b), lq, lk, dk]
        
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        k = torch.unsqueeze(k, dim=1) # [(n*b), 1, lk, dk]
        k = k.expand(k.shape[0], len_q, k.shape[2], k.shape[3]) # [(n*b), lq, lk, dk]
        
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
        
        mask = mask.repeat(n_head, 1, 1) # (n*b) x lq x lk
        
        ## Map based Attention
        #output, attn = self.attention(q, k, v, mask=mask)
        q_k = torch.cat([q, k], dim=3) # [(n*b), lq, lk, dk * 2]
        attn = self.weight_map(q_k).squeeze(dim=3) # [(n*b), lq, lk]
        
        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]
        attn = self.dropout(attn) # [n * b, l_q, l_k]
        
        # [n * b, l_q, l_k] * [n * b, l_v, d_v] >> [n * b, l_q, d_v]
        output = torch.bmm(attn, v)
        
        output = output.view(n_head, sz_b, len_q, d_v)
        
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.act(self.fc(output)))
        output = self.layer_norm(output + residual)

        return output, attn
    
def expand_last_dim(x, num):
    view_size = list(x.size()) + [1]
    expand_size = list(x.size()) + [num]
    return x.view(view_size).expand(expand_size)

class AttnModel(torch.nn.Module):
    """Attention based temporal layers
    """
    def __init__(self, feat_dim, edge_dim, time_dim, 
                 attn_mode='prod', n_head=2, drop_out=0.1):
        """
        args:
          feat_dim: dim for the node features
          edge_dim: dim for the temporal edge features
          time_dim: dim for the time encoding
          attn_mode: choose from 'prod' and 'map'
          n_head: number of heads in attention
          drop_out: probability of dropping a neural.
        """
        super(AttnModel, self).__init__()
        
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        
        self.edge_in_dim = (feat_dim + edge_dim + time_dim)
        self.model_dim = self.edge_in_dim
        #self.edge_fc = torch.nn.Linear(self.edge_in_dim, self.feat_dim, bias=False)

        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)

        #self.act = torch.nn.ReLU()
        
        assert(self.model_dim % n_head == 0)
        self.logger = logging.getLogger(__name__)
        self.attn_mode = attn_mode
        
        if attn_mode == 'prod':
            self.multi_head_target = MultiHeadAttention(n_head, 
                                             d_model=self.model_dim, 
                                             d_k=self.model_dim // n_head, 
                                             d_v=self.model_dim // n_head, 
                                             dropout=drop_out)
            self.logger.info('Using scaled prod attention')
            
        elif attn_mode == 'map':
            self.multi_head_target = MapBasedMultiHeadAttention(n_head, 
                                             d_model=self.model_dim, 
                                             d_k=self.model_dim // n_head, 
                                             d_v=self.model_dim // n_head, 
                                             dropout=drop_out)
            self.logger.info('Using map based attention')
        else:
            raise ValueError('attn_mode can only be prod or map')
        
        
    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        """"Attention based temporal attention forward pass
        args:
          src: float Tensor of shape [B, D]
          src_t: float Tensor of shape [B, Dt], Dt == D
          seq: float Tensor of shape [B, N, D]
          seq_t: float Tensor of shape [B, N, Dt]
          seq_e: float Tensor of shape [B, N, De], De == D
          mask: boolean Tensor of shape [B, N], where the true value indicate a null value in the sequence.

        returns:
          output, weight

          output: float Tensor of shape [B, D]
          weight: float Tensor of shape [B, N]
        """

        src_ext = torch.unsqueeze(src, dim=1) # src [B, 1, D]
        src_e_ph = torch.zeros_like(src_ext)
        q = torch.cat([src_ext, src_e_ph, src_t], dim=2) # [B, 1, D + De + Dt] -> [B, 1, D]
        k = torch.cat([seq, seq_e, seq_t], dim=2) # [B, 1, D + De + Dt] -> [B, 1, D]
        
        mask = torch.unsqueeze(mask, dim=2) # mask [B, N, 1]
        mask = mask.permute([0, 2, 1]) #mask [B, 1, N]

        # # target-attention
        # print(q.dtype,k.dtype,mask.dtype)
        output, attn = self.multi_head_target(q=q, k=k, v=k, mask=mask) # output: [B, 1, D + Dt], attn: [B, 1, N]

        output = output.squeeze()

        attn = attn.squeeze()

        output = self.merger(output, src)

        return output, attn


class TGAN(torch.nn.Module):
    def __init__(self, ngh_finders, n_feat, e_feat,
                 attn_mode='prod', use_time='time', agg_method='attn', 
                 num_layers=3, n_head=4, null_idx=0, drop_out=0.1, seq_len=None):
        """
        init arguments:

        --n_head N_HEAD                     number of heads used in attention layer
        --num_layers N_LAYER                number of network layers
        --drop_out DROP_OUT                 dropout probability
        --agg_method {attn,lstm,mean}       local aggregation method
                                
        --attn_mode {prod,map}              use dot product attention or mapping based
                                
        --use_time {time,pos,empty}         how to use time information
                                
        --n_feat                            raw features of nodes
        --e_feat                            raw edge features of links
        """
        super(TGAN, self).__init__()
        self.train_ngh_finder = ngh_finders[0]
        self.full_ngh_finder = ngh_finders[1]

        self.num_neighbors = seq_len
        self.num_layers = num_layers 
        self.ngh_finder = self.train_ngh_finder
        self.null_idx = null_idx
        self.logger = logging.getLogger(__name__)

        self.n_feat_th = torch.nn.Parameter(n_feat)
        self.edge_raw_embed = torch.nn.Embedding.from_pretrained(e_feat.float(), padding_idx=0, freeze=True)
        #freeze=true相当于训练的时候不更新edge_raw_embed
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(n_feat.float(), padding_idx=0, freeze=True)
        #raw_embed在训练时不更新
        
        self.feat_dim = n_feat.shape[1]
        
        self.n_feat_dim = self.feat_dim
        self.e_feat_dim = self.feat_dim
        self.model_dim = self.feat_dim
        
        self.use_time = use_time
       
        if agg_method == 'attn':
            self.logger.info('Aggregation uses attention model')
            self.attn_model_list = torch.nn.ModuleList([AttnModel(self.feat_dim, 
                                                               self.feat_dim, 
                                                               self.feat_dim,
                                                               attn_mode=attn_mode, 
                                                               n_head=n_head, 
                                                               drop_out=drop_out) for _ in range(num_layers)])
        elif agg_method == 'lstm':
            self.logger.info('Aggregation uses LSTM model')
            self.attn_model_list = torch.nn.ModuleList([LSTMPool(self.feat_dim,
                                                                 self.feat_dim,
                                                                 self.feat_dim) for _ in range(num_layers)])
        elif agg_method == 'mean':
            self.logger.info('Aggregation uses constant mean model')
            self.attn_model_list = torch.nn.ModuleList([MeanPool(self.feat_dim,
                                                                 self.feat_dim) for _ in range(num_layers)])
        else:
        
            raise ValueError('invalid agg_method value, use attn or lstm')
        
        
        if use_time == 'time':
            self.logger.info('Using time encoding')
            self.time_encoder = TimeEncode(expand_dim=self.n_feat_dim)
        elif use_time == 'pos':
            assert(seq_len is not None)
            self.logger.info('Using positional encoding')
            self.time_encoder = PosEncode(expand_dim=self.n_feat_dim, seq_len=seq_len)
        elif use_time == 'empty':
            self.logger.info('Using empty encoding')
            self.time_encoder = EmptyEncode(expand_dim=self.n_feat_dim)
        else:
            raise ValueError('invalid time option!')
    
    def set_state(self, state):
        if state != "train" and state !="eval":
            raise ValueError("training mode is expected to be <train> or <eval>")
        if state == "train":
            self.ngh_finder = self.train_ngh_finder
            self.train()
        else :
            self.ngh_finder = self.full_ngh_finder
            self.eval()

    def forward(self, node_idxs, cut_time, num_neighbors=None):
        if num_neighbors == None:
            num_neighbors = self.num_neighbors

        node_embed = self.tem_conv(node_idxs, cut_time, self.num_layers, num_neighbors)

        return node_embed


    def tem_conv(self, src_idx_l, cut_time_l, curr_layers, num_neighbors=20):
        assert(curr_layers >= 0)
        
        device = self.n_feat_th.device
    
        batch_size = len(src_idx_l)
        
        src_node_batch_th = torch.from_numpy(src_idx_l).long().to(device)
        cut_time_l_th = torch.from_numpy(cut_time_l).float().to(device)
        
        cut_time_l_th = torch.unsqueeze(cut_time_l_th, dim=1)

        # query node always has the start time -> time span == 0
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_l_th))
        
        if curr_layers == 0:
            src_node_feat = self.node_raw_embed(src_node_batch_th)
            return src_node_feat
        else:
            src_node_conv_feat = self.tem_conv(src_idx_l, 
                                           cut_time_l,
                                           curr_layers=curr_layers - 1, 
                                           num_neighbors=num_neighbors)
            
            
            src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = self.ngh_finder.get_temporal_neighbor( 
                                                                    src_idx_l, 
                                                                    cut_time_l, 
                                                                    num_neighbors)
            
            src_ngh_node_batch_th = torch.from_numpy(src_ngh_node_batch).long().to(device)
            src_ngh_eidx_batch = torch.from_numpy(src_ngh_eidx_batch).long().to(device)
            
            src_ngh_t_batch_delta = cut_time_l[:, np.newaxis] - src_ngh_t_batch
            src_ngh_t_batch_th = torch.from_numpy(src_ngh_t_batch_delta).float().to(device)

            # get previous layer's node features
            src_ngh_node_batch_flat = src_ngh_node_batch.flatten() #reshape(batch_size, -1)
            src_ngh_t_batch_flat = src_ngh_t_batch.flatten() #reshape(batch_size, -1)  
            src_ngh_node_conv_feat = self.tem_conv(src_ngh_node_batch_flat, 
                                                   src_ngh_t_batch_flat,
                                                   curr_layers=curr_layers - 1, 
                                                   num_neighbors=num_neighbors)
            src_ngh_feat = src_ngh_node_conv_feat.view(batch_size, num_neighbors, -1)
            
            # get edge time features and node features
            src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)
            src_ngn_edge_feat = self.edge_raw_embed(src_ngh_eidx_batch)

            # attention aggregation
            mask = src_ngh_node_batch_th == 0
            attn_m = self.attn_model_list[curr_layers - 1]



            # print(src_node_conv_feat.dtype, src_node_t_embed.dtype, src_ngh_feat.dtype, src_ngh_t_embed.dtype, src_ngn_edge_feat.dtype, mask.dtype)
            local, weight = attn_m(src_node_conv_feat, 
                                   src_node_t_embed,
                                   src_ngh_feat,
                                   src_ngh_t_embed, 
                                   src_ngn_edge_feat, 
                                   mask)
            return local
        # assert(curr_layers >= 0)
        
        # device = self.n_feat_th.device
    
        # batch_size = len(src_idx_l)
    
        # src_node_batch_th = src_idx_l.long().to(device)
        # cut_time_l_th = cut_time_l.float().to(device)

        # cut_time_l_th = torch.unsqueeze(cut_time_l_th, dim=1)

        # # query node always has the start time -> time span == 0
        # src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_l_th))

        # if curr_layers == 0:
        #     src_node_feat = self.node_raw_embed(src_node_batch_th)
        #     # print("src_node_feat",type(src_node_feat))
        #     return src_node_feat
        # else:
        #     src_node_conv_feat = self.tem_conv(src_idx_l, 
        #                                    cut_time_l,
        #                                    curr_layers=curr_layers - 1, 
        #                                    num_neighbors=num_neighbors)
            
        #     src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = self.ngh_finder.get_temporal_neighbor( 
        #                                                             src_idx_l, 
        #                                                             cut_time_l, 
        #                                                             num_neighbors=num_neighbors)
            
        #     #输入目标节点，在他的所有任意时间节点的邻居中选择num_neighbors个邻居并按照链接时间排序
        #     src_ngh_node_batch_th = src_ngh_node_batch.long().to(device)
        #     src_ngh_eidx_batch = src_ngh_eidx_batch.long().to(device)
            
        #     src_ngh_t_batch_delta = cut_time_l.view(-1,1) - src_ngh_t_batch
        #     src_ngh_t_batch_th = src_ngh_t_batch_delta.float().to(device)
        #     #节点的时间与采样的邻居时间的差


        #     # get previous layer's node features
        #     src_ngh_node_batch_flat = src_ngh_node_batch.flatten() #reshape(batch_size, -1)
        #     src_ngh_t_batch_flat = src_ngh_t_batch.flatten() #reshape(batch_size, -1)  
        #     src_ngh_node_conv_feat = self.tem_conv(src_ngh_node_batch_flat, 
        #                                            src_ngh_t_batch_flat,
        #                                            curr_layers=curr_layers - 1, 
        #                                            num_neighbors=num_neighbors)
        #     # print(src_ngh_node_conv_feat)
        #     src_ngh_feat = src_ngh_node_conv_feat.view(batch_size, num_neighbors, -1)
            
        #     # get edge time features and node features
        #     src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)
        #     # time features

        #     src_ngn_edge_feat = self.edge_raw_embed(src_ngh_eidx_batch)
        #     # time features

        #     # attention aggregation
        #     mask = src_ngh_node_batch_th == 0
        #     attn_m = self.attn_model_list[curr_layers - 1]
        #     # print(src_node_conv_feat.dtype, src_node_t_embed.dtype, src_ngh_feat.dtype, src_ngh_t_embed.dtype, src_ngn_edge_feat.dtype, mask.dtype)

        #     # print("src_node_conv_feat ",src_node_conv_feat.shape,src_node_conv_feat.dtype)

        #     # print("src_node_t_embed",src_node_t_embed.shape,src_node_t_embed.dtype)

        #     # print("src_ngh_feat ",src_ngh_feat.shape,src_ngh_feat.dtype)

        #     # print("src_ngh_t_embed",src_ngh_t_embed.shape,src_ngh_t_embed.dtype)

        #     # print("src_ngn_edge_feat",src_ngn_edge_feat.shape,src_ngn_edge_feat.dtype)

        #     # print("mask",mask.shape,mask.dtype)
            
        #     local, weight = attn_m(src_node_conv_feat, 
        #                            src_node_t_embed,
        #                            src_ngh_feat,
        #                            src_ngh_t_embed, 
        #                            src_ngn_edge_feat, 
        #                            mask)
        #     return local
        