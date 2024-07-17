import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from model.base import *
from model.rnn import RnnDecoder
from model.induced_att import ContinuousEncoding


def get_batch_mask(B, L, valid_len):
    mask = repeat(torch.arange(end=L, device=valid_len.device),
                  'L -> B L', B=B) >= repeat(valid_len, 'B -> B L', L=L)  # (B, L)
    return mask


class PositionalEncoding(nn.Module):
    """
    A type of trigonometric encoding for indicating items' positions in sequences.
    """

    def __init__(self, embed_size, max_len):
        super().__init__()

        pe = torch.zeros(max_len, embed_size).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, position_ids=None):
        """
        Args:
            x: (B, T, d_model)
            position_ids: (B, T) or None

        Returns:
            (1, T, d_model) / (B, T, d_model)
        """
        if position_ids is None:
            return self.pe[:, :x.size(1)]
        else:
            batch_size, seq_len = position_ids.shape
            pe = self.pe[:, :seq_len, :]  # (1, T, d_model)
            pe = pe.expand((position_ids.shape[0], -1, -1))  # (B, T, d_model)
            pe = pe.reshape(-1, self.d_model)  # (B * T, d_model)
            position_ids = position_ids.reshape(-1, 1).squeeze(1)  # (B * T,)
            output_pe = pe[position_ids].reshape(batch_size, seq_len, self.d_model).detach()
            return output_pe


class PositionalEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super(PositionalEmbedding, self).__init__()

        self.hidden_size = hidden_size

        inv_freq = 1 / (10000 ** (torch.arange(0.0, hidden_size, 2.0) / hidden_size))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        B, L = pos_seq.shape
        sinusoid_inp = torch.ger(rearrange(pos_seq, 'B L -> (B L)'), self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        pos_emb = rearrange(pos_emb, '(B L) E -> B L E', B=B, L=L)

        return pos_emb


class MLMTransformer(Encoder):
    def __init__(self, sampler, d_model, output_size, dis_feats, num_embeds, con_feats,
                 token_feat, num_tokens, seq_feat, pool_type,
                 num_heads=8, num_layers=2, hidden_size=128):
        super().__init__(sampler, 'MLMTransformer-' + ','.join(map(str, dis_feats + con_feats)) +
                         f'-d{d_model}-h{hidden_size}-l{num_layers}-h{num_heads}-o{output_size}-pool{pool_type}')

        self.token_feat = token_feat
        self.num_tokens = num_tokens
        self.seq_feat = seq_feat
        self.pool_type = pool_type

        self.dis_feats = dis_feats
        self.con_feats = con_feats
        self.num_heads = num_heads
        self.output_size = output_size

        transformer_layer = nn.TransformerEncoderLayer(d_model, num_heads, hidden_size, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers)

        self.token_embed = nn.Embedding(num_tokens + 1, d_model)
        self.dis_embeds = nn.ModuleList([nn.Embedding(num_embed, d_model) for num_embed in num_embeds])
        self.con_embeds = nn.ModuleList([ContinuousEncoding(d_model) for _ in con_feats])
        self.seq_encode = ContinuousEncoding(d_model)

        self.out_linear = nn.Sequential(nn.Linear(d_model, output_size, bias=False),
                                        nn.LayerNorm(output_size),
                                        nn.LeakyReLU(inplace=True),
                                        nn.Linear(output_size, output_size))

    def forward(self, trip, valid_len, pretrain=False):
        B, L, E_in = trip.shape

        src_key_padding_mask = repeat(torch.arange(end=L, device=trip.device),
                                      'L -> B L', B=B) >= repeat(valid_len, 'B -> B L', L=L)  # (B, L)
        if pretrain:
            token = trip[..., self.token_feat].long()  # (B, L)
            src_mask = repeat((token < self.num_tokens), 'B L2 -> (B N) L1 L2', N=self.num_heads, L1=L)
            src_mask = src_mask & repeat(TransformerDecoder.gen_casual_mask(L).to(trip.device),
                                         'L1 L2 -> (B N) L1 L2', B=B, N=self.num_heads)
        else:
            token = (torch.ones(B, L).to(trip.device) * self.num_tokens).long()
            src_mask = None

        dis_x = torch.stack([embed(trip[..., feat].long()).masked_fill_((token < self.num_tokens).unsqueeze(-1), 0)
                             for feat, embed in zip(self.dis_feats, self.dis_embeds)], -1).sum(-1)
        con_x = torch.stack([embed(trip[..., feat].float()).masked_fill_((token < self.num_tokens).unsqueeze(-1), 0)
                             for feat, embed in zip(self.con_feats, self.con_embeds)], -1).sum(-1)
        token_x = self.token_embed(token)
        seq_enc = self.seq_encode(trip[..., self.seq_feat].long())  # (B, L, d_model)

        out = self.transformer(dis_x + con_x + token_x + seq_enc, mask=src_mask,
                               src_key_padding_mask=src_key_padding_mask)
        out = self.out_linear(out)
        if not pretrain:
            out = out.mean(1)
        return out


class DualPosTransformer(Encoder):
    def __init__(self, sampler, d_model, output_size, num_heads=8, num_layers=2, hidden_size=128):
        super().__init__(sampler, 'DualPosTransformer-' +
                         f'-d{d_model}-h{hidden_size}-l{num_layers}-h{num_heads}-o{output_size}')

        transformer_layer = nn.TransformerEncoderLayer(d_model, num_heads, hidden_size, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers)

        self.pos_embeds = nn.ModuleList([PositionalEmbedding(d_model) for _ in range(2)])
        self.out_linear = nn.Sequential(nn.Linear(d_model, output_size, bias=False),
                                        nn.LayerNorm(output_size),
                                        nn.LeakyReLU(inplace=True),
                                        nn.Linear(output_size, output_size))

    def forward(self, trip, trip_len, pos=None, src_mask=None, batch_mask=None):
        B, L_trip, _ = trip.shape
        if batch_mask is None:
            batch_mask = get_batch_mask(B, L_trip, trip_len)
        pos_x = torch.stack([embed(pos[..., i]) for i, embed in enumerate(self.pos_embeds)], -1).sum(-1)

        out = self.transformer(trip + pos_x, mask=src_mask, src_key_padding_mask=batch_mask)
        out = self.out_linear(out)

        return out


class TransformerEncoder(Encoder):
    def __init__(self, d_model, output_size,
                 sampler, dis_feats=[], num_embeds=[], con_feats=[],
                 num_heads=8, num_layers=2, hidden_size=128):
        super().__init__(sampler, 'Transformer-' +
                         ','.join(map(str, dis_feats + con_feats)) +
                         f'-d{d_model}-h{hidden_size}-l{num_layers}-h{num_heads}')

        self.d_model = d_model
        self.dis_feats = dis_feats
        self.con_feats = con_feats
        self.output_size = output_size

        self.pos_encode = PositionalEncoding(d_model, max_len=2000)
        if len(dis_feats):
            assert len(dis_feats) == len(num_embeds), \
                'length of num_embeds list should be equal to the number of discrete features.'
            self.dis_embeds = nn.ModuleList([nn.Embedding(num_embed, d_model) for num_embed in num_embeds])
        else:
            self.dis_embeds = None

        if len(con_feats):
            self.con_linear = nn.Linear(len(con_feats), d_model)
        else:
            self.con_linear = None

        self.out_linear = nn.Sequential(nn.Linear(d_model, output_size, bias=False),
                                        nn.LayerNorm(output_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(output_size, output_size))

        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, hidden_size, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, trip, valid_len):
        B, L, E_in = trip.shape

        src_mask = repeat(torch.arange(end=L, device=trip.device),
                          'L -> B L', B=B) >= repeat(valid_len, 'B -> B L', L=L)  # (B, L)
        x, src_mask = self.sampler(trip, src_mask)

        h = torch.zeros(B, x.size(1), self.d_model).to(x.device)
        if self.dis_embeds is not None:
            for dis_embed, dis_feat in zip(self.dis_embeds, self.dis_feats):
                h += dis_embed(x[..., dis_feat].long())  # (B, L, E)
        if self.con_linear is not None:
            h += self.con_linear(x[..., self.con_feats])
        h += self.pos_encode(h)

        memory = self.encoder(h, src_key_padding_mask=src_mask)  # (B, L, E)
        memory = torch.nan_to_num(memory)
        mask_expanded = repeat(src_mask, 'B L -> B L E', E=memory.size(2))  # (B, L, E)
        memory = memory.masked_fill(mask_expanded, 0)  # (B, L, E)
        memory = torch.sum(memory, 1) / valid_len.unsqueeze(-1)
        memory = self.out_linear(memory)  # (B, E_out) or (B, L, E_out)
        return memory


class TransformerDecoder(Decoder):
    def __init__(self, encode_size, d_model, hidden_size, num_layers, num_heads):
        super().__init__(f'TransDecoder-d{d_model}-h{hidden_size}-l{num_layers}-h{num_heads}')
        self.d_model = d_model

        self.memory_linear = nn.Linear(encode_size, d_model)
        self.pos_encode = PositionalEncoding(d_model, max_len=2000)
        self.start_token = nn.Parameter(torch.randn(d_model), requires_grad=True)

        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads,
                                           dim_feedforward=hidden_size, batch_first=True)
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)

    def forward(self, tgt, encode):
        memory = self.memory_linear(encode).unsqueeze(1)  # (B, 1, E)

        tgt_mask = self.gen_casual_mask(tgt.size(1)).to(tgt.device)
        out = self.transformer(tgt + self.pos_encode(tgt), memory, tgt_mask=tgt_mask)
        return out

    @staticmethod
    def gen_casual_mask(seq_len, include_self=True):
        """
        Generate a casual mask which prevents i-th output element from
        depending on any input elements from "the future".
        Note that for PyTorch Transformer model, sequence mask should be
        filled with -inf for the masked positions, and 0.0 else.

        :param seq_len: length of sequence.
        :return: a casual mask, shape (seq_len, seq_len)
        """
        if include_self:
            mask = 1 - torch.triu(torch.ones(seq_len, seq_len)).transpose(0, 1)
        else:
            mask = 1 - torch.tril(torch.ones(seq_len, seq_len)).transpose(0, 1)
        return mask.bool()


class TransformerDenoiser(Denoiser):
    def __init__(self, input_size, y_size, d_model, hidden_size, num_layers, num_heads):
        super().__init__('TransDenoiser' + f'-d{d_model}-h{hidden_size}-l{num_layers}-h{num_heads}')

        self.input_linear = nn.Linear(input_size, d_model)
        self.y_linear = nn.Linear(y_size, d_model)
        self.time_embed = TimeEmbed(d_model, d_model)

        self.pos_encode = PositionalEncoding(d_model, 500)
        self.transformer = nn.Transformer(d_model=d_model, nhead=num_heads,
                                          num_encoder_layers=num_layers, num_decoder_layers=num_layers,
                                          dim_feedforward=hidden_size, dropout=0.1, batch_first=True)

        output_size = input_size
        self.out_linear = nn.Sequential(nn.Linear(d_model, output_size, bias=False),
                                        nn.LayerNorm(output_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(output_size, output_size))

    def forward(self, x, t, y, valid_len=None):
        x = self.input_linear(x)  # (B, L, E)
        t = self.time_embed(t).unsqueeze(1)
        y = self.y_linear(y)
        if len(y.shape) < 3:
            y = y.unsqueeze(1)

        src = t + y  # (B, 1, E)
        tgt = x + self.pos_encode(x)
        tgt_mask = None
        if valid_len is not None:
            B, L, _ = x.shape
            tgt_mask = repeat(torch.arange(end=L, device=x.device),
                              'L -> B L', B=B) >= repeat(valid_len, 'B -> B L', L=L)  # (B, L)
        out = self.transformer(src, tgt, tgt_key_padding_mask=tgt_mask)

        out = self.out_linear(out)
        return out
