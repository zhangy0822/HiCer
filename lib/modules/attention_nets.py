from torch.nn import functional as F
# from models.transformer.utils import PositionWiseFeedForward
import torch
from torch import nn
# from models.transformer.attention import MultiHeadAttention
from lib.modules.containers import Module
import numpy as np

class PositionWiseFeedForward(nn.Module):
    '''
    Position-wise feed forward layer
    '''

    def __init__(self, d_model=512, d_ff=2048, dropout=.1):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input):

        out = self.fc2(self.dropout_2(F.relu(self.fc1(input))))
        out = self.dropout(out)
        out = self.layer_norm(input + out)
        return out


class ScaledDotProductAttentionMemory(nn.Module):
    '''
    Scaled dot-product attention with memory
    '''

    def __init__(self, d_model, d_k, d_v, h):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param m: Number of memory slots
        '''
        super(ScaledDotProductAttentionMemory, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_gk = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_gv = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        # self.m_k = nn.Parameter(torch.FloatTensor(1, m, h * d_k))
        # self.m_v = nn.Parameter(torch.FloatTensor(1, m, h * d_v))

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        # self.m = m

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.xavier_uniform_(self.fc_gk.weight)
        nn.init.xavier_uniform_(self.fc_gv.weight)
        # nn.init.normal_(self.m_k, 0, 1 / self.d_k)
        # nn.init.normal_(self.m_v, 0, 1 / self.m)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)
        nn.init.constant_(self.fc_gk.bias, 0)
        nn.init.constant_(self.fc_gv.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, key_grid=None, value_grid=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        # m_k = np.sqrt(self.d_k) * self.m_k.expand(b_s, self.m, self.h * self.d_k)
        # m_v = np.sqrt(self.m) * self.m_v.expand(b_s, self.m, self.h * self.d_v)
        if len(key_grid.size()) == 2:
            key_grid = key_grid.unsqueeze(1)
            value_grid = value_grid.unsqueeze(1)

        k = self.fc_k(keys)
        gk = self.fc_gk(key_grid)
        k_new = torch.cat([k,gk],dim=1)
        k_new = k_new.view(b_s, k_new.size(1), self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(values)
        gv = self.fc_gv(key_grid)
        v_new = torch.cat([v,gv],dim=1)
        v_new = v_new.view(b_s, v_new.size(1), self.h, self.d_v).permute(0, 2, 1, 3)

        # keys_new = torch.cat([keys, key_grid], 1)
        # values_new = torch.cat([values, value_grid], 1)
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        # k = self.fc_k(keys_new).view(b_s, keys_new.size(1), self.h, self.d_k).permute(0, 2, 3, 1)
        # v = self.fc_k(values_new).view(b_s, keys_new.size(1), self.h, self.d_k).permute(0, 2, 1, 3)
        # k = torch.cat([self.fc_k(keys), m_k], 1).view(b_s, nk + self.m, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        # v = torch.cat([self.fc_v(values), m_v], 1).view(b_s, nk + self.m, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k_new) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        if attention_mask is not None:
            att[:, :, :, :nk] = att[:, :, :, :nk].masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        out = torch.matmul(att, v_new).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

class MultiHeadAttention(Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        super(MultiHeadAttention, self).__init__()


        self.attention = ScaledDotProductAttentionMemory(d_model=d_model, d_k=d_k, d_v=d_v, h=h)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)


    def forward(self, queries, keys, values, attention_mask=None, key_grid=None, value_grid=None):

        out = self.attention(queries, keys, values, attention_mask, key_grid, value_grid)
        out = self.dropout(out)
        out = self.layer_norm(queries + out)

        return out


class EncoderLayer(nn.Module):
    def __init__(self, d_model=1024, d_k=128, d_v=128, h=8, d_ff=2048, dropout=.1):
        super(EncoderLayer, self).__init__()

        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, queries, keys, values, attention_mask=None, key_grid=None):
        att = self.mhatt(queries, keys, values, attention_mask, key_grid, key_grid)
        ff = self.pwff(att)
        return ff


class GridAugmentedEncoder(nn.Module):
    def __init__(self, d_in=2048, padding_idx=0, d_model=1024, d_k=128, d_v=128, h=8, d_ff=2048, dropout=.1):
        super(GridAugmentedEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.fc = nn.Linear(d_in, self.d_model)
        # self.fc_g = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        self.layers = EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout)
            
        self.padding_idx = padding_idx

    def forward(self, input, input_grid):

        # input (b_s, seq_len, d_in)
        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)
        
        # out_g = F.relu(self.fc_g(input_grid))
        # out_g = self.dropout(out_g)
        # out_g = self.layer_norm(out_g) 
             
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        out = self.layers(out, out, out, attention_mask, input_grid)

        return out


