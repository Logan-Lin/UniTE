import torch
import torch.nn as nn

from modules.model.base import *


class RobustDAALayer(nn.Module):
    '''
    The layer of RobustDAA
    '''
    def __init__(self, input_size, hidden_size, dropout=0):
        super(RobustDAALayer, self).__init__()
        self.linear = nn.Linear(input_size,hidden_size)
        self.sigmoid = nn.Sigmoid()
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
    

class RobustDAA_Attention(RobustDAALayer):
    '''
    The attention module of RobustDAA
    '''
    def __init__(self, input_size, hidden_size, dropout=0):
        super().__init__(input_size, hidden_size, dropout)
        self.name = f'RobustDAA_Attention-i{input_size}-o{hidden_size}'
    
    def forward(self, src_XR, tgt_XR, L_D, S):
        x = self.linear(L_D)
        x = self.sigmoid(x)
        return x


class RobustDAA_Encoder(Encoder):
    """
    RobustDAA Encoder.
    Zhang, Rui, et al. "Clustering noisy trajectories via robust deep attention auto-encoders." 2019 20th IEEE International Conference on Mobile Data Management (MDM). IEEE, 2019.
    """
    def __init__(self, input_size, hidden_decrease, num_layers, sampler, dropout=0, grid_col=None, aux_cols=None):
        super().__init__(sampler, f'RobustDAA_Encoder-d{input_size}-hidden_dec{hidden_decrease}-l{num_layers}')
        assert num_layers > 1 , "The number of encoder layers must be larger than 1"
        self.input_size = input_size
        self.num_layers = num_layers

        self.encoder = nn.Sequential()
        self.encoder.add_module("hidden_layer1", RobustDAALayer(input_size, input_size, dropout))
        self.output_size = input_size
        for i in range(1, self.num_layers):
            name = 'hidden_layer' + str(i+1)
            self.encoder.add_module(name, RobustDAALayer(self.output_size, self.output_size-hidden_decrease, dropout))
            self.output_size -= hidden_decrease
    
    def forward(self, src_XR, tgt_XR, L_D, S):
        x = self.encoder(L_D)
        return x
    

class RobustDAA_Decoder(Decoder):
    def __init__(self, input_size, hidden_increase, enc_num_layers, dropout=0, grid_col=None, aux_cols=None):
        super().__init__(f'RobustDAA_Decoder-d{input_size}-hidden_inc{hidden_increase}-l{enc_num_layers-1}')
        self.input_size = input_size
        self.num_layers = enc_num_layers-1
        
        self.decoder = nn.Sequential()
        self.output_size = input_size
        for i in range(self.num_layers):
            name = 'hidden_layer' + str(i+1)
            self.decoder.add_module(name,RobustDAALayer(self.output_size, self.output_size+hidden_increase, dropout))
            self.output_size += hidden_increase

    def forward(self, x):
        x = self.decoder(x)
        return x
