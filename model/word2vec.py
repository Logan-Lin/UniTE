import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from model.base import *


class GeoConstrainSkipGramEncoder(Encoder):
    def __init__(self, d_model, output_size, sampler,
                 dis_feats=[], num_embeds=[]):
        super().__init__(sampler, f'GeoSkipgram-d{d_model}'+
                         ','.join(map(str, dis_feats)))

        assert not d_model % 2, "Embedding dim should be divided by 2."
        embed_dim = d_model // 2
        self.d_model = d_model
        self.dis_feats = dis_feats
        self.output_size = output_size

        assert len(dis_feats) == 1, "The number of embedding feature can be only road id"
        assert len(dis_feats) == len(num_embeds), \
            "length of num_embeds list should be equal to the number of discrete features."

        self.s_embeddings = nn.Embedding(num_embeds[0], embed_dim, sparse=False)  # When sparse is True, you must use SparseAdam
        self.d_embeddings = nn.Embedding(num_embeds[0], embed_dim, sparse=False)
        # Use lookup table and output embedding at the same time, or comment the following two lines
        self.v_s_embeddings = nn.Embedding(num_embeds[0], embed_dim, sparse=False)
        self.v_d_embeddings = nn.Embedding(num_embeds[0], embed_dim, sparse=False)

        self.out_linear = nn.Sequential(nn.Linear(d_model, output_size, bias=False),
                                        nn.LayerNorm(output_size),
                                        nn.LeakyReLU(inplace=True),
                                        nn.Linear(output_size, output_size))

        initrange = 1.0 / embed_dim
        init.uniform_(self.s_embeddings.weight.data, -initrange, initrange)
        init.uniform_(self.d_embeddings.weight.data, -initrange, initrange)

    def forward_train(self, pos_s, pos_d, pos_m, neg_v):
        emb_s = self.s_embeddings(pos_s)  # (bs, d)
        emb_d = self.d_embeddings(pos_d)  # (bs, d)

        # Use only lookup table
        # emb_ms = self.s_embeddings(pos_m)  # (bs, d)
        # emb_md = self.d_embeddings(pos_m)  # (bs, d)
        # emb_neg_vs = self.s_embeddings(neg_v)  # (bs, num_neg, d)
        # emb_neg_vd = self.d_embeddings(neg_v)  # (bs, num_neg, d)

        # Use lookup table and output embedding at the same time
        emb_ms = self.v_s_embeddings(pos_m)  # (bs, d)
        emb_md = self.v_d_embeddings(pos_m)  # (bs, d)
        emb_neg_vs = self.v_s_embeddings(neg_v)  # (bs, num_neg, d)
        emb_neg_vd = self.v_d_embeddings(neg_v)  # (bs, num_neg, d)

        emb_sd = torch.concat([emb_s, emb_d], dim=1)
        emb_m = torch.concat([emb_ms, emb_md], dim=1)
        emb_neg = torch.concat([emb_neg_vs, emb_neg_vd], dim=2)

        return emb_sd, emb_m, emb_neg

    def forward(self, trip, valid_len):
        # trip: (bs, l, d)
        B, L, E_in = trip.shape

        src_key_padding_mask = repeat(torch.arange(end=L, device=trip.device),
                                      'L -> B L ', B=B) >= repeat(valid_len, 'B -> B L', L=L)  # (B, L)

        emb_s = self.s_embeddings(trip[:, :, self.dis_feats[0]].long())
        emb_d = self.d_embeddings(trip[:, :, self.dis_feats[0]].long())
        trip_embed = torch.concat([emb_s, emb_d], dim=2)

        src_key_padding_mask = repeat(src_key_padding_mask, 'B L -> B L E', E=trip_embed.shape[2])
        valid_len = repeat(valid_len, 'B -> B E', E=trip_embed.shape[2])

        trip_embed = self.out_linear(trip_embed)  # extra linear which not exists in the source paper
        trip_embed = trip_embed.masked_fill_(~src_key_padding_mask, 0)
        trip_embed = trip_embed.sum(dim=1) / valid_len

        return trip_embed
