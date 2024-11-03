import math

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import repeat, rearrange


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal-based function used for encoding timestamps.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimeEmbed(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, time):
        return self.time_mlp(time)


class Encoder(nn.Module):
    def __init__(self, sampler, name):
        super().__init__()

        self.sampler = sampler
        self.name = f'{name}-{sampler.name}'


class Decoder(nn.Module):
    def __init__(self, name):
        super().__init__()

        # self.denormalizer = denormalizer
        self.name = name


class Denoiser(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name


class GNN(nn.Module):
    def __init__(self, graph, name):
        super().__init__()

        self.graph = graph

        self.name = name