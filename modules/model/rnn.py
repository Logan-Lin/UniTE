from torch.nn.utils.rnn import pack_padded_sequence

from modules.model.base import *


class RnnEncoder(Encoder):
    """
    A basic RNN-based Encoder.
    """
    def __init__(self, rnn_type, num_embed, d_model, hidden_size, num_layers, output_size,
                 variational, sampler, road_col=None, aux_cols=None):
        super().__init__(sampler, f'RNN-{rnn_type}-' + str(road_col) +
                         (''.join(map(str, aux_cols)) if aux_cols is not None else '') +
                         f'-var{int(variational)}-d{d_model}-h{hidden_size}-l{num_layers}-o{output_size}')
        self.road_col = road_col
        self.aux_cols = aux_cols
        self.d_model = d_model
        self.output_size = output_size
        self.variational = variational

        if self.road_col is not None:
            self.road_embed = nn.Embedding(num_embed, d_model)
        else:
            self.road_embed = None

        if aux_cols is not None and len(aux_cols) > 0:
            self.aux_linear = nn.Linear(len(aux_cols), d_model)
        else:
            self.aux_linear = None

        rnn_params = {"input_size": d_model, "hidden_size": hidden_size, "num_layers": num_layers,
                      "batch_first": True, "dropout": 0.1}
        self.rnn = nn.GRU(**rnn_params) if rnn_type == 'gru' else nn.RNN(**rnn_params)

        if variational:
            self.norm_linear = nn.Linear(hidden_size, output_size * 2)
        else:
            self.out_linear = nn.Linear(hidden_size, output_size)

    def forward(self, trip, valid_len):
        B, L = trip.size(0), trip.size(1)

        valid_mask = repeat(torch.arange(end=L, device=trip.device),
                            'L -> B L', B=B) < repeat(valid_len, 'B -> B L', L=L)  # (B, L)
        x, valid_mask = self.sampler(trip, valid_mask)
        valid_len = valid_mask.long().sum(-1).long().to('cpu')  # (B)

        h = torch.zeros(B, x.size(1), self.d_model).to(x.device)
        if self.road_embed is not None:
            h += self.road_embed(x[:, :, self.road_col].long())  # (B, L, E)
        if self.aux_linear is not None:
            h += self.aux_linear(x[:, :, self.aux_cols])

        h = pack_padded_sequence(h, lengths=valid_len, batch_first=True, enforce_sorted=False)
        _, h = self.rnn(h)
        h = h[-1]  # (B, hidden_size)

        if self.variational:
            mean, var = self.norm_linear(h).chunk(2, -1)
            noise = torch.randn_like(mean)
            h = var * noise + mean
        else:
            h = self.out_linear(h)  # (B, output_size)
        return h


class RnnDecoder(Decoder):
    """
    A basic RNN-based Decoder.
    """
    def __init__(self, rnn_type, num_roads, encode_size,
                 d_model, hidden_size, num_layers, road_col=None, aux_cols=None):
        super().__init__(f'RNN-{rnn_type}-' + str(road_col) +
                         (''.join(map(str, aux_cols)) if aux_cols is not None else '') +
                         f'-d{d_model}-h{hidden_size}-l{num_layers}')
        self.road_col = road_col
        self.aux_cols = aux_cols
        self.d_model = d_model
        self.num_layers = num_layers

        self.init_linear = nn.Linear(encode_size, num_layers * hidden_size)

        rnn_params = {"input_size": d_model, "hidden_size": hidden_size, "num_layers": num_layers,
                      "batch_first": True, "dropout": 0.1}
        self.rnn = nn.GRU(**rnn_params) if rnn_type == 'gru' else nn.RNN(**rnn_params)

        self.start_token = nn.Parameter(torch.randn(d_model), requires_grad=True)
        if road_col is not None:
            self.road_embed = nn.Embedding(num_roads, d_model)
            self.road_predictor = nn.Linear(hidden_size, num_roads)
        else:
            self.road_embed = None
            self.road_predictor = None

        if aux_cols is not None and len(aux_cols):
            self.aux_linear = nn.Linear(len(aux_cols), d_model)
            self.aux_predictor = nn.Linear(hidden_size, len(aux_cols))
        else:
            self.aux_linear = None
            self.aux_predictor = None

    def forward(self, trip, valid_len, encode):
        B, L = trip.size(0), trip.size(1)
        valid_len = valid_len.long().to('cpu')

        x = trip
        init_h = rearrange(self.init_linear(encode), 'B (D H) -> D B H', D=self.num_layers).contiguous()
        init_c = torch.zeros(init_h.shape, device=init_h.device).contiguous()

        h = torch.zeros(B, x.size(1), self.d_model).to(x.device)   # (B, L, E)
        if self.road_embed is not None:
            h += self.road_embed(x[:, :, self.road_col].long())
        if self.aux_linear is not None:
            h += self.aux_linear(x[:, :, self.aux_cols])

        tgt = torch.cat([repeat(self.start_token, 'E -> B 1 E', B=B),
                         h[:, :-1]], 1)  # (B, L, E), the target sequence
        out, _ = self.rnn(tgt, init_h)  # (B * L, hidden_size)

        predict, label = [], []
        loss = 0.0

        if self.road_predictor is not None:
            road_pre, road_label, road_loss = self.get_pre_label(x, valid_len, self.road_col,
                                                                 out, self.road_predictor, True)
            predict.append(road_pre)
            label.append(road_label)
            loss += road_loss

        if self.aux_predictor is not None:
            aux_pre, aux_label, aux_loss = self.get_pre_label(x, valid_len, self.aux_cols,
                                                              out, self.aux_predictor, False)
            predict.append(aux_pre)
            label.append(aux_label)
            loss += aux_loss

        return np.concatenate(label, -1), np.concatenate(predict, -1), loss

    @staticmethod
    def get_pack_data(x, valid_len):
        return pack_padded_sequence(x, valid_len.long().to('cpu'), batch_first=True, enforce_sorted=False).data

    @staticmethod
    def get_pre_label(x, valid_len, col, out, predictor, discrete):
        pre = predictor(out)
        label = x[..., col]
        flat_pre = RnnDecoder.get_pack_data(pre, valid_len)
        flat_label = RnnDecoder.get_pack_data(label, valid_len)
        if discrete:
            label = label.long()
            return pre.argmax(-1).long().detach().cpu().numpy(), \
                label.long().detach().cpu().numpy(), \
                F.cross_entropy(flat_pre, flat_label.long())
        else:
            return pre.detach().cpu().numpy(), \
                label.detach().cpu().numpy(), \
                F.mse_loss(flat_pre, flat_label)
