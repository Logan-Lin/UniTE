import numpy as np
from dgl.nn.pytorch import conv
import torch
from torch import nn
from einops import rearrange

from model.base import GNN, TimeEmbed


class GATDenoiser(GNN):
    def __init__(self, graph, input_size, y_size, d_model, **params):
        super().__init__(graph, 'GATDenoiser')

        self.time_embed = TimeEmbed(d_model, d_model)
        self.input_linear = nn.Sequential(nn.Linear(input_size, d_model), nn.SiLU())
        self.y_linear = nn.Sequential(nn.Linear(y_size, d_model), nn.SiLU())
        self.gnn = conv.GATConv(in_feats=d_model, out_feats=d_model // params['num_heads'], **params)
        self.res_linear = nn.Sequential(nn.Linear(d_model * 2, d_model),
                                        nn.SiLU(), nn.Linear(d_model, input_size))

    def forward(self, x, t, y):
        x = self.input_linear(x)  # (B, N, E_model)
        t = self.time_embed(t).unsqueeze(1)  # (B, 1, E_model)
        y = self.y_linear(y).unsqueeze(1)  # (B, 1, E_model)

        x = x + t + y
        h = self.gnn(self.graph, rearrange(x, 'B N E -> N B E'))  # (N, B, N_head, E_model)
        h = rearrange(h, 'N B H E -> B N (H E)')  # (B, N, E_model)
        h = torch.cat([x, h], -1)  # (B, N, 2*E_model)
        h = self.res_linear(h)  # (B, N, E_in)
        return h


class GCNDenoiser(GNN):
    def __init__(self, graph, input_size, y_size, d_model, **params):
        super().__init__(graph, 'GCNDenoiser')

        self.time_embed = TimeEmbed(d_model, d_model)
        self.input_linear = nn.Sequential(nn.Linear(input_size, d_model), nn.SiLU())
        self.y_linear = nn.Sequential(nn.Linear(y_size, d_model), nn.SiLU())
        self.gnn = conv.GCN2Conv(in_feats=d_model, layer=1, **params)
        self.res_linear = nn.Sequential(nn.Linear(d_model * 2, d_model),
                                        nn.SiLU(), nn.Linear(d_model, input_size))

    def forward(self, x, t, y):
        x = self.input_linear(x)  # (B, N, E_model)
        t = self.time_embed(t).unsqueeze(1)  # (B, 1, E_model)
        y = self.y_linear(y).unsqueeze(1)  # (B, 1, E_model)

        x = x + t + y
        feat = rearrange(x, 'B N E -> N B E')
        h = self.gnn(self.graph, feat, feat)  # (N, B, N_head, E_model)
        h = rearrange(h, 'N B E -> B N E')  # (B, N, E_model)
        h = torch.cat([x, h], -1)  # (B, N, 2*E_model)
        h = self.res_linear(h)  # (B, N, E_in)
        return h