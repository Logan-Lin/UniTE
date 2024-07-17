import numpy as np
import pickle
import math
import pandas
import torch
from torch import nn
from sklearn.cluster import *
from sklearn import preprocessing
import random


class Trajectory2VecEncoder(nn.Module):
    def __init__(self, sampler, input_size, hidden_size):
        super().__init__()
        self.sampler = sampler
        self.name = f'Trajectory2Vec_encoder_{input_size}_{hidden_size}'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = hidden_size
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, trip, valid_len, *para):
        x = self.sampler(trip, valid_len)
        output, (ht, ct) = self.encoder(x)
        return ht.permute(1, 0, 2)


class Trajectory2vecDecoder(nn.Module):
    def __init__(self, sampler, input_size, hidden_size, seq_num, device):
        super().__init__()
        self.name = f'Trajectory2Vec_decoder_{input_size}_{hidden_size}_{seq_num}'
        self.sampler = sampler
        self.device = device
        self.seq_num = seq_num
        self.hidden_size = hidden_size
        self.output_size = hidden_size
        self.gates = nn.Linear(hidden_size * 2, hidden_size * 4)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        self.linear = nn.Linear(hidden_size, input_size)
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, trip, valid_len, embedding):
        x = self.sampler(trip, valid_len)
        batch_size = embedding.size(0)
        embedding_seq_len = embedding.size(1)
        h = torch.zeros(batch_size, embedding_seq_len, self.output_size).to(self.device)
        c = torch.zeros(batch_size, embedding_seq_len, self.hidden_size).to(self.device)
        y_list = []
        c_list = []
        for i in range(self.seq_num):
            if i == 0:
                forget_gate, input_gate, output_gate, candidate_cell = \
                    self.gates(torch.cat([embedding[:, i:i + 1, :], h[:, i:i + 1, :]], dim=-1)).chunk(4, -1)
                forget_gate, input_gate, output_gate = (self.sigmoid(g)
                                                        for g in (forget_gate, input_gate, output_gate))
            else:
                forget_gate, input_gate, output_gate, candidate_cell = \
                    self.gates(torch.cat([c, h], dim=-1)).chunk(4, -1)
                forget_gate, input_gate, output_gate = (self.sigmoid(g)
                                                        for g in (forget_gate, input_gate, output_gate))
            c = forget_gate * c + input_gate * self.tanh(candidate_cell)
            h = output_gate * self.tanh(c)
            c_list.append(c)
            y_list.append(self.output(h))
        c_out = torch.cat(c_list, dim=-2)
        out = self.linear(c_out)
        return x, out
