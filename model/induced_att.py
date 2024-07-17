from model.base import *


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

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class ContinuousEncoding(nn.Module):
    """
    A type of trigonometric encoding for encode continuous values into distance-sensitive vectors.
    """

    def __init__(self, embed_size):
        super().__init__()
        self.omega = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, embed_size))).float(),
                                  requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(embed_size).float(), requires_grad=True)
        self.div_term = math.sqrt(1. / embed_size)

    def forward(self, x):
        """
        :param x: input sequence for encoding, (batch_size, seq_len)
        :return: encoded sequence, shape (batch_size, seq_len, embed_size)
        """
        encode = x.unsqueeze(-1) * self.omega.reshape(1, 1, -1) + self.bias.reshape(1, 1, -1)
        encode = torch.cos(encode)
        return self.div_term * encode


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True, share_weight=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.share_weight = share_weight
        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv_proj = nn.ModuleList([nn.Linear(embed_dim, embed_dim, bias=bias) for _ in range(3)])
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.temperature = embed_dim ** 0.5

    def forward(self, query, key, value, key_padding_mask=None):
        query, key, value = (self.qkv_proj[i](item).reshape(item.size(0), item.size(1), self.num_heads, self.head_dim).transpose(0, 2)
                             for i, item in zip((0, 0, 1) if self.share_weight else (0, 1, 2), (query, key, value)))  # (num_heads, batch_size, seq_len, head_dim)
        attn_weight = torch.matmul(query / self.temperature, key.transpose(2, 3)
                                   )  # (num_heads, batch_size, query_len, kv_len)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.reshape(1, key_padding_mask.size(
                0), 1, key_padding_mask.size(1))  # (1, batch_size, 1, kv_len)
            attn_weight = attn_weight.masked_fill(key_padding_mask, float('-inf'))
        attn_weight = self.dropout(torch.nan_to_num(torch.softmax(attn_weight, -1), nan=0.0))

        output = torch.matmul(attn_weight, value)  # (num_heads, batch_size, query_len, head_dim)
        output = output.transpose(0, 2).reshape(output.size(
            2), -1, self.embed_dim)  # (query_len, batch_size, embed_dim)

        return output, attn_weight


class FeedForwardLayer(nn.Module):
    """The position-wise feed forward layer."""

    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)

    def forward(self, x):
        x = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        return x


class IAEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, bias=True):
        """
        :param d_model: the number of expected features in the input.
        :param nhead: the number of heads in the multi-head attention layer.
        :param dim_feedforward: the dimension of the feedforward network model.
        :param use_ffn: whether to apply feed-forward layer.
        :param dropout: a Dropout layer applied on the output weights of attention layer and the feed-forward layer.
        """
        super().__init__()

        self.attn = MultiHeadAttention(d_model, nhead, dropout=dropout, bias=bias)
        self.ffn = FeedForwardLayer(d_model, dim_feedforward, dropout=dropout)

        self.layer_norm = nn.ModuleList([nn.LayerNorm(d_model, eps=1e-6) for _ in range(2)])
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)])

    def forward(self, input_seq, anchor, key_padding_mask=None):
        """
        :param input_seq: sequences of input features, shape (seq_len, batch_size, d_model)
        :param anchor: the anchor sequence for summarizing the input sequence, shape (anchor_len, batch_size, d_model)
        :param key_padding_mask: if provided, specified padding elements in the key will be ignored by the attention,
            shape (batch_size, seq_len). Expected to be a Boolean tensor, where True positions will be ignored.
        :param attn_mask: 2D or 3D mask that prevents attention to certain positions, shape (anchor_len, seq_len) or
            (batch_size * num_heads, anchor_len, seq_len). Expected to be a Boolean tensor, where True positions will be ignored.
        :return: attention output, shape (anchor_len, batch_size, d_model).
        """
        residual = anchor
        output, weight = self.attn(anchor, input_seq, input_seq,
                                   key_padding_mask=key_padding_mask)
        output = self.dropout[0](output) + residual
        output = self.layer_norm[0](output)

        residual = output
        output = self.ffn(output)
        output = self.dropout[1](output) + residual
        output = self.layer_norm[1](output)
        return output


class InducedAttEncoder(Encoder):
    def __init__(self, d_model, dis_feats, num_embeds, con_feats, hidden_size, num_heads, output_size,
                 anchor_length, sampler):
        super().__init__(sampler, 'IA-' + ''.join(map(str, dis_feats + con_feats)) +
                         f'-d{d_model}-h{hidden_size}-l2-a{anchor_length}-h{num_heads}-o{output_size}')

        self.dis_feats = dis_feats
        self.con_feats = con_feats
        self.d_model = d_model
        self.output_size = output_size

        num_last_anchor = output_size // d_model
        # self.num_anchor_list = [num_last_anchor * 2 ** n for n in range(num_layers-1, 0, -1)] + [num_last_anchor]
        self.num_anchor_list = [anchor_length, num_last_anchor] if anchor_length > 0 else [num_last_anchor]
        self.encoder_layers = nn.ModuleList([IAEncoderLayer(d_model, num_heads, hidden_size)
                                             for _ in range(len(self.num_anchor_list))])

        if dis_feats is not None and len(dis_feats) > 0:
            assert len(dis_feats) == len(num_embeds), \
                'length of num_embeds list should be equal to the number of discrete features.'
            self.dis_embeds = nn.ModuleList([nn.Embedding(num_embed, d_model) for num_embed in num_embeds])
        else:
            self.dis_embeds = None

        if con_feats is not None and len(con_feats) > 0:
            self.con_linear = nn.Linear(len(con_feats), d_model)
        else:
            self.con_linear = None

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.anchors = nn.ModuleList([nn.Embedding(num_anchor, d_model) for num_anchor in self.num_anchor_list])
        self.pos_encode = PositionalEncoding(d_model, max_len=500)

        self.out_linear = nn.Sequential(nn.Linear(output_size, output_size, bias=False),
                                        nn.BatchNorm1d(output_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(output_size, output_size, bias=False),
                                        nn.BatchNorm1d(output_size))

    def forward(self, trip, valid_len):
        B, L = trip.size(0), trip.size(1)

        src_mask = repeat(torch.arange(end=L, device=trip.device),
                          'L -> B L', B=B) >= repeat(valid_len, 'B -> B L', L=L)  # (B, L)
        x, src_mask = self.sampler(trip, src_mask)

        # Apply positional encoding.
        h = torch.zeros(B, x.size(1), self.d_model).to(x.device)
        if self.dis_embeds is not None:
            for dis_embed, dis_feat in zip(self.dis_embeds, self.dis_feats):
                h += dis_embed(x[:, :, dis_feat].long())  # (B, L, E)
        h += self.pos_encode(h)
        if self.con_linear is not None:
            h = h + self.con_linear(x[:, :, self.con_feats])
        h = rearrange(h, 'B L E -> L B E')

        for i, layer in enumerate(self.encoder_layers):
            h = layer(h, self.anchors[i].weight.unsqueeze(1).repeat(1, B, 1),
                      key_padding_mask=src_mask if i == 0 else None)

        h = rearrange(self.layer_norm(h), 'A B E -> B (A E)')  # (B, num_last_anchor * d_model)
        h = self.out_linear(h)
        return h
