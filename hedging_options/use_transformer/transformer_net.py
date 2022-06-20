# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/17
Description:
"""
import os
import sys

import torch
import torch.nn as nn

# Append the library path to PYTHONPATH, so library can be imported.
sys.path.append(os.path.dirname(os.getcwd()))


class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, dropout, device, max_length=100):
        super().__init__()

        self.device = device

        # self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.pos = None
        self.layers = nn.ModuleList([EncoderLayer(input_dim, hid_dim, n_heads, dropout, device) for _ in range(
            n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src):
        # src = [batch size, src len]
        # src_mask = [batch size, 1, 1, src len]
        # batch_size = src.shape[0]
        # src_len = src.shape[1]
        # src_feature_len = src.shape[2]

        # if self.pos is None:
        # self.pos = torch.arange(0, src_len).repeat(src_feature_len, 1).transpose(0, 1).repeat(batch_size, 1,
        #                                                                                       1).float().to(self.device)

        # pos = [batch size, src len]

        # src = self.dropout(src * self.scale + self.pos)

        # src = [batch size, src len, hid dim]
        for layer in self.layers:
            src = layer(src)

        # src = [batch size, src len, hid dim]

        return src


class EncoderLayer(nn.Module):
    def __init__(self, input_dim, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(input_dim)
        self.ff_layer_norm = nn.LayerNorm(input_dim)
        self.self_attention = MultiHeadAttentionLayer(input_dim, hid_dim, n_heads, dropout, device)
        # self.self_attention_2 = MultiHeadAttentionLayer(hid_dim,input_dim, n_heads, dropout, device)
        # self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, 1, 1, src len]

        # self attention
        # print(src.shape)
        _src, _ = self.self_attention(src, src, src)
        # __src, __ = self.self_attention_2(_src, _src, _src)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        # positionwise feedforward
        # _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        # src = self.ff_layer_norm(src + self.dropout(_src))
        # src = self.ff_layer_norm(src)

        # src = [batch size, src len, hid dim]

        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, input_dim, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(input_dim, hid_dim)
        self.fc_k = nn.Linear(input_dim, hid_dim)
        self.fc_v = nn.Linear(input_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, input_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x, attention


# class PositionwiseFeedforwardLayer(nn.Module):
#     def __init__(self, hid_dim, pf_dim, dropout):
#         super().__init__()
#
#         self.fc_1 = nn.Linear(hid_dim, pf_dim)
#         self.fc_2 = nn.Linear(pf_dim, hid_dim)
#
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         # x = [batch size, seq len, hid dim]
#
#         x = self.dropout(torch.relu(self.fc_1(x)))
#
#         # x = [batch size, seq len, pf dim]
#
#         x = self.fc_2(x)
#
#         # x = [batch size, seq len, hid dim]
#
#         return x


class Seq2Seq(nn.Module):
    def __init__(self, encoder, input_dim, device):
        super().__init__()

        self.encoder = encoder
        # self.decoder = None
        # self.src_pad_idx = src_pad_idx
        # self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.fc_o = nn.Linear(input_dim, 1)
        self.fc_o2 = nn.Linear(15, 64)
        self.fc_o3 = nn.Linear(64, 1)

    def forward(self, src, results):
        # src = [batch size, src len]
        # trg = [batch size, trg len]

        # src_mask = self.make_src_mask(src)
        # trg_mask = self.make_trg_mask(trg)

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]
        # self.fc_o2(torch.transpose(self.encoder(src),1,2))
        # output_ = torch.relu(self.fc_o(self.encoder(src)))
        # output = self.fc_o2(output_.view(output_dim, -1))
        output = self.fc_o3(torch.relu(self.fc_o2(torch.transpose(self.encoder(src), 1, 2)[:, -1, :])))
        # output = self.fc_o2(torch.transpose(self.encoder(src), 1, 2)).view(src.shape[0], -1)[:, -3].reshape((
        #     src.shape[0],1))
        output_dim = output.shape[0]

        if torch.isnan(output).any():
            print('\nweight error')
        # print(output.shape)
        # v_1 = output.contiguous().view(output_dim)
        # i = 0
        # for t in src:
        #     i += 1
        #     if torch.isnan(t).any():
        #         print(i)

        R_1 = results.contiguous().view(output_dim, -1)

        # print(output.shape)
        # print(R_1.shape)

        # output = output - R_1[:, 1].view(output_dim, -1)

        # print(R_1[:, 1].shape)
        # print(output.shape)
        s_1 = R_1[:, -1].view(output_dim, -1)
        one_ret = 1 + R_1[:, 0].view(output_dim, -1) / 253
        s_0 = R_1[:, 3].view(output_dim, -1)
        c_0 = R_1[:, 2].view(output_dim, -1)

        output = output * s_1 + one_ret * (c_0 - output * s_0)

        # print(output.shape)
        # enc_src = [batch size, src len, hid dim]

        # output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]

        return output

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        # nn.init.xavier_uniform_(m.weight.data)
        nn.init.kaiming_normal_(m.weight.data)
