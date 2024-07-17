from torch.nn.utils.rnn import pack_padded_sequence

from model.base import *


class Traj2VecEncoder(Encoder):
    def __init__(self, num_embed, d_model, hidden_size, num_layers, output_size, sampler,
                 pre_embed=None, pre_embed_update=True):
        super().__init__(sampler, f'Traj2Vec-' +
                         f'-d{d_model}-h{hidden_size}-l{num_layers}-o{output_size}')
        self.d_model = d_model+1
        self.output_size = output_size

        self.road_embed = nn.Embedding(num_embed, d_model)
        if pre_embed is not None:
            self.road_embed.weight = nn.Parameter(torch.from_numpy(pre_embed),
                                                  requires_grad=pre_embed_update)

        rnn_params = {"input_size": d_model+1, "hidden_size": hidden_size, "num_layers": num_layers,
                      "batch_first": True, "dropout": 0.1}
        self.rnn = nn.LSTM(**rnn_params)

        self.out_linear = nn.Linear(hidden_size, output_size)

    def forward(self, trip, valid_len, time_diff):
        B, L = trip.size(0), trip.size(1)

        valid_mask = repeat(torch.arange(end=L, device=trip.device),
                            'L -> B L', B=B) < repeat(valid_len, 'B -> B L', L=L)  # (B, L)
        x, valid_mask = self.sampler(trip, valid_mask)
        valid_len = valid_mask.long().sum(-1).long().to('cpu')  # (B)

        h = torch.zeros(B, x.size(1), self.d_model).to(x.device)
        if self.road_embed is not None:
            h[:, :, :-1] += self.road_embed(x.long())  # (B, L, E)
            h[:, :, -1] += time_diff  # (B, L)

        h = pack_padded_sequence(h, lengths=valid_len, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.rnn(h)
        h = h[-1]  # (B, hidden_size)

        h = self.out_linear(h)  # (B, output_size)

        return h


class Traj2VecDecoder(Decoder):
    def __init__(self, encode_size,
                 d_model, hidden_size, num_layers):
        super().__init__(f'Traj2Vec-' +
                         f'-d{d_model}-h{hidden_size}-l{num_layers}')
        self.d_model = d_model
        self.num_layers = num_layers

        self.init_linear = nn.Linear(encode_size, num_layers * hidden_size)

        rnn_params = {"input_size": d_model, "hidden_size": hidden_size, "num_layers": num_layers,
                      "batch_first": True, "dropout": 0.1}
        self.rnn = nn.LSTM(**rnn_params)

        self.start_token = nn.Parameter(torch.randn(d_model), requires_grad=True)

    def forward(self, tgt, encode):
        B = len(tgt)

        init_h = rearrange(self.init_linear(encode), 'B (D H) -> D B H', D=self.num_layers).contiguous()
        init_c = torch.zeros(init_h.shape, device=init_h.device).contiguous()

        out, _ = self.rnn(tgt, (init_h, init_c))  # (B * L, hidden_size)

        return out
