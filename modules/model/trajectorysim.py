import os

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import h5py

from modules.model.base import *


class StackingGRUCell(nn.Module):
    """
    Multi-layer CRU Cell
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(StackingGRUCell, self).__init__()
        self.num_layers = num_layers
        self.grus = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        self.grus.append(nn.GRUCell(input_size, hidden_size))
        for i in range(1, num_layers):
            self.grus.append(nn.GRUCell(hidden_size, hidden_size))

    def forward(self, input, h0):
        """
        Input:
        input (batch, input_size): input tensor
        h0 (num_layers, batch, hidden_size): initial hidden state
        ---
        Output:
        output (batch, hidden_size): the final layer output tensor
        hn (num_layers, batch, hidden_size): the hidden state of each layer
        """
        hn = []
        output = input
        for i, gru in enumerate(self.grus):
            hn_i = gru(output, h0[i])
            hn.append(hn_i)
            if i != self.num_layers - 1:
                output = self.dropout(hn_i)
            else:
                output = hn_i
        hn = torch.stack(hn)
        return output, hn


class GlobalAttention(nn.Module):
    """
    $$a = \sigma((W_1 q)H)$$
    $$c = \tanh(W_2 [a H, q])$$
    """
    def __init__(self, hidden_size):
        super(GlobalAttention, self).__init__()
        self.L1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.L2 = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, q, H):
        """
        Input:
        q (batch, hidden_size): query
        H (batch, seq_len, hidden_size): context
        ---
        Output:
        c (batch, hidden_size)
        """
        # (batch, hidden_size) => (batch, hidden_size, 1)
        q1 = self.L1(q).unsqueeze(2)
        # (batch, seq_len)
        a = torch.bmm(H, q1).squeeze(2)
        a = self.softmax(a)
        # (batch, seq_len) => (batch, 1, seq_len)
        a = a.unsqueeze(1)
        # (batch, hidden_size)
        c = torch.bmm(a, H).squeeze(1)
        # (batch, hidden_size * 2)
        c = torch.cat([c, q], 1)
        return self.tanh(self.L2(c))


# 在TranjectorySim中，encoder和decoder共用一个Embedding层
class TrajSimEmbed(nn.Module):
    def __init__(self, meta_dir, meta_type, d_model, PAD, pretrain):
        super(TrajSimEmbed, self).__init__()
        self.dist_path = os.path.join(meta_dir, meta_type + '_dist.h5')
        if pretrain:
            with h5py.File(self.dist_path, "r") as f:
                v = f["V"][...]
            self.vocab_size = len(v)
        else:
            self.vocab_size = 0
        self.name = f'TrajSimEmbed-i{self.vocab_size}-o{d_model}'
        self.embedding = nn.Embedding(self.vocab_size, d_model, padding_idx=PAD)

    def forward(self, input):
        """
        Input:  input (seq_len, batch): padded sequence tensor
        Output: (seq_len, batch, input_size)
        """
        return self.embedding(input)

class TrajSimEncoder(Encoder):
    """
    TrajectorySim Encoder.
    Chen, Ziwen, et al. "Towards robust trajectory similarity computation: Representation-based spatio-temporal similarity quantification." World Wide Web 26.4 (2023): 1271-1294.
    """
    # d_model -> embedding_size/input_size
    # num_layers = 3
    # dropout = args.dropout = 0.2
    # bidirectional = true
    def __init__(self, num_embed, d_model, hidden_size, num_layers, dropout,
                       bidirectional, sampler, grid_col=None, aux_cols=None):
        super().__init__(sampler, f'TrajSimEncoder-d{d_model}-h{hidden_size}-l{num_layers}')
        self.num_directions = 2 if bidirectional else 1
        assert hidden_size % self.num_directions == 0
        self.hidden_size = hidden_size // self.num_directions
        self.output_size = self.hidden_size * self.num_directions
        self.num_layers = num_layers
        self.grid_col = grid_col
        self.embedding = nn.Embedding(num_embed, d_model)  # for downstream task
        self.rnn = nn.GRU(d_model, self.hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          dropout=dropout)

    # 将encoder的输出隐藏层转化为decoder的隐藏层输入
    def encoder_hn2decoder_h0(self, h):
        """
        Input:
        h (num_layers * num_directions, batch, hidden_size): encoder output hn
        ---
        Output:
        h (num_layers, batch, hidden_size * num_directions): decoder input h0
        """
        if self.num_directions == 2:
            num_layers, batch, hidden_size = h.size(0) // 2, h.size(1), h.size(2)
            return h.view(num_layers, 2, batch, hidden_size) \
                .transpose(1, 2).contiguous() \
                .view(num_layers, batch, hidden_size * 2)
        else:
            return h
    
    def load_pretrained_embedding(self, path):
        if os.path.isfile(path):
            w = torch.load(path)
            self.embedding.weight.data.copy_(w)

    def forward(self, input, lengths, trg=None, h0=None, embed_layer=None):
        """
        Input:
        input (batch, seq_len): padded sequence tensor
        lengths (batch): sequence lengths
        h0 (num_layers*num_directions, batch, hidden_size): initial hidden state
        embed_layer: TranjectorySim Embedding shared with decoder, only used in pretrain
        ---
        Output:
        hn (num_layers*num_directions, batch, hidden_size):
            the hidden state of each layer
        output (seq_len, batch, hidden_size*num_directions): output tensor
        """
        B, L = input.size(0), input.size(1)
        valid_mask = repeat(torch.arange(end=L, device=input.device),
                            'L -> B L', B=B) < repeat(lengths, 'B -> B L', L=L)  # (B, L)
        x, valid_mask = self.sampler(input, valid_mask)
        lengths = valid_mask.long().sum(-1).long().to("cpu")  # (B)

        x = x.transpose(0, 1)
        # the embedding layer for data in pre-training and downstream task is different
        # (seq_len, batch) => (seq_len, batch, input_size)
        if embed_layer is not None:  # pretrain
            embed = embed_layer(x.long())
        else:  # downstream task
            x = x[:, :, self.grid_col]
            embed = self.embedding(x.long())

        lengths = lengths.data.view(-1).tolist()
        if lengths is not None:
            embed = pack_padded_sequence(embed, lengths, enforce_sorted=False)
        output, hn = self.rnn(embed, h0)

        if embed_layer is not None:  # pretrain
            if lengths is not None:
                output = pad_packed_sequence(output)[0]
            return hn, output
        else:
            # take the last layer as representations (downstream task)
            return self.encoder_hn2decoder_h0(hn)[-1]


class TrajSimDecoder(Decoder):
    # d_model -> embedding_size/input_size
    def __init__(self, d_model, hidden_size, num_layers, dropout, grid_col=None, aux_cols=None):
        super().__init__(f'TrajSimDecoder-d{d_model}-h{hidden_size}-l{num_layers}')
        self.rnn = StackingGRUCell(d_model, hidden_size, num_layers,
                                   dropout)
        self.attention = GlobalAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # self.grid_col = grid_col

    def forward(self, input, h, H, embed_layer, use_attention=True):
        """
        Input:
        input (batch, seq_len): padded sequence tensor
        h (num_layers, batch, hidden_size): input hidden state
        H (seq_len, batch, hidden_size): the context used in attention mechanism
            which is the output of encoder
        use_attention: If True then we use attention
        embed_layer: TranjectorySim Embedding shared with decoder
        ---
        Output:
        output (seq_len, batch, hidden_size)
        h (num_layers, batch, hidden_size): output hidden state,
            h may serve as input hidden state for the next iteration,
            especially when we feed the word one by one (i.e., seq_len=1)
            such as in translation
        """
        # if len(input.shape) == 3:
        #     input = input[:, :, self.grid_col]
        input = input.transpose(0, 1)
        # (seq_len, batch) => (seq_len, batch, input_size)
        embed = embed_layer(input.long())
        output = []
        # split along the sequence length dimension
        for e in embed.split(1):
            e = e.squeeze(0) # (1, batch, input_size) => (batch, input_size)
            o, h = self.rnn(e, h)
            if use_attention:
                o = self.attention(o, H.transpose(0, 1))
            o = self.dropout(o)
            output.append(o)
        output = torch.stack(output)
        return output, h
