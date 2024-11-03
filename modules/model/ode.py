import torch
import torchcde
from torchdiffeq import odeint_adjoint as odeint

from modules.model.base import *
from modules.model.cnf_layers import ODECNFfunc, CNF
from modules.model.induced_att import IAEncoderLayer, PositionalEncoding


class CDEEncoder(Encoder):
    def __init__(self, input_cols, hidden_size, output_size, sampler):
        super().__init__(sampler, 'CDE-' + ','.join(map(str, input_cols)) + 
                         f'-h{hidden_size}-o{output_size}')

        self.input_cols = input_cols
        self.input_size = len(input_cols)
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.net = nn.Linear(hidden_size, self.input_size * hidden_size)
        self.z_start = nn.Parameter(torch.rand(hidden_size).float(), requires_grad=True)
        self.out_linear = nn.Linear(hidden_size * 2, output_size)

    def cde_neural(self, t, z):
        return rearrange(self.net(z), 'B (H I) -> B H I', H=self.hidden_size, I=self.input_size)

    def forward(self, trip, valid_len):
        B, L, E_in = trip.shape

        x, = self.sampler(trip)
        x = x[..., self.input_cols]

        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
        X = torchcde.CubicSpline(coeffs)
        z0 = self.z_start.unsqueeze(0).repeat(B, 1)
        z = torchcde.cdeint(X=X, func=self.cde_neural, z0=z0, t=X.interval)  # (B, 2, hidden_size)

        z = rearrange(z, 'B Z H -> B (Z H)')
        z = self.out_linear(z)
        return z


class CoeffAttEncoder(Encoder):
    def __init__(self, input_cols, d_model, hidden_size, output_size, anchor_length, num_heads, sampler):
        super().__init__(sampler, 'CoA-' + ''.join(map(str, input_cols)) +
                         f'-d{d_model}-h{hidden_size}-l2-a{anchor_length}-h{num_heads}-o{output_size}')

        self.input_cols = input_cols
        self.input_size = len(input_cols) * 4
        self.d_model = d_model
        self.output_size = output_size

        self.input_linear = nn.Sequential(nn.Linear(self.input_size, d_model, bias=False),
                                          nn.LayerNorm(d_model, eps=1e-6), nn.SiLU(inplace=True))

        num_last_anchor = output_size // d_model
        # self.num_anchor_list = [num_last_anchor * 2 ** n for n in range(num_layers-1, 0, -1)] + [num_last_anchor]
        self.num_anchor_list = [anchor_length, num_last_anchor] if anchor_length > 0 else [num_last_anchor]
        self.encoder_layers = nn.ModuleList([IAEncoderLayer(d_model, num_heads, hidden_size)
                                             for _ in range(len(self.num_anchor_list))])
        self.anchors = nn.ModuleList([nn.Embedding(num_anchor, d_model) for num_anchor in self.num_anchor_list])
        self.pos_encode = PositionalEncoding(d_model, max_len=500)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.out_linear = nn.Sequential(nn.Linear(output_size, output_size, bias=False),
                                        nn.BatchNorm1d(output_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(output_size, output_size, bias=False),
                                        nn.BatchNorm1d(output_size))

    def forward(self, trip, valid_len):
        B, L, _ = trip.shape
        src_mask = repeat(torch.arange(end=L, device=trip.device),
                          'L -> B L', B=B) >= repeat(valid_len-1, 'B -> B L', L=L)  # (B, L)
        x, src_mask = self.sampler(trip, src_mask)
        x = x[:, :, self.input_cols]  # (B, L, E)

        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)  # (B, L, E*4)
        src_mask = src_mask[:, :coeffs.size(1)]

        h = self.input_linear(coeffs)  # (B, L, d_model)
        h += self.pos_encode(h)
        h = rearrange(h, 'B L E -> L B E')

        for i, layer in enumerate(self.encoder_layers):
            h = layer(h, self.anchors[i].weight.unsqueeze(1).repeat(1, B, 1),
                      key_padding_mask=src_mask if i == 0 else None)

        h = rearrange(self.layer_norm(h), 'A B E -> B (A E)')  # (B, num_last_anchor * d_model)
        h = self.out_linear(h)
        return h


class STODEEncoder(Encoder):
    """
    Spatiotemporal ODE Encoder used in TrajODE.
    Liang, Yuxuan, et al. "Modeling Trajectories with Neural Ordinary Differential Equations." IJCAI. 2021.
    """
    def __init__(self, input_cols, rnn_hidden_size, ode_hidden_size, cnf_width, output_size, sampler, ode_flag=True):
        super().__init__(sampler, 'STODE')

        self.input_cols = input_cols
        self.input_size = len(input_cols)
        self.output_size = output_size

        self.ode_flag = ode_flag

        self.inital_state = nn.Parameter(torch.zeros(rnn_hidden_size), requires_grad=True)
        if ode_flag:
            self.gru_cell = nn.GRUCell(self.input_size, rnn_hidden_size)
            self.ode_func = self.ODEFunc(rnn_hidden_size, ode_hidden_size)
            # self.cnf = self.CNF(d_model=rnn_hidden_size, hidden_size=ode_hidden_size, width=cnf_width)
            self.st_gate = nn.Sequential(nn.Linear(self.input_size, rnn_hidden_size),
                                         nn.ReLU())
        else:
            self.gru = nn.GRU(self.input_size, rnn_hidden_size, num_layers=2, batch_first=True)

        self.out_linear = nn.Sequential(nn.Linear(rnn_hidden_size, rnn_hidden_size, bias=False),
                                        nn.BatchNorm1d(rnn_hidden_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(rnn_hidden_size, output_size, bias=False),
                                        nn.BatchNorm1d(output_size))
        # self.p_z0 = torch.distributions.MultivariateNormal(
        #     loc=torch.zeros(rnn_hidden_size).float(),
        #     covariance_matrix=torch.diag(torch.ones(rnn_hidden_size) * 0.1).float())

    def forward(self, trip, valid_len):
        B, L, _ = trip.shape
        x, = self.sampler(trip)
        x = x[:, :, self.input_cols]  # (B, L, E)

        t = torch.arange(0, L+1).float().to(x.device)

        if self.ode_flag:
            h = self.inital_state.unsqueeze(0).repeat(B, 1)  # (B, rnn_hidden_size)
            for i in range(L):
                # Calculate the candidate hidden state through RNN cell and ODE solver.
                h_cdd_gru = self.gru_cell(x[:, i], h)  # (B, hidden_size)
                h = odeint(self.ode_func, h_cdd_gru, torch.stack([t[i], t[i+1]]), method='euler')[-1]

        else:
            h = self.inital_state.unsqueeze(0).unsqueeze(0).repeat(2, B, 1)  # (B, rnn_hidden_size)
            _, h = self.gru(x, h)
            h = h[-1]

        out = self.out_linear(h)
        return out

    class ODEFunc(nn.Module):
        def __init__(self, d_model, hidden_size):
            super().__init__()

            self.net = nn.Sequential(nn.Linear(d_model, hidden_size),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(hidden_size, hidden_size),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(hidden_size, d_model))

        def forward(self, t, x):
            return self.net(x)


class TrajODEDecoder(Decoder):
    """
    Trajectory ODE Decoder.
    """
    def __init__(self, encode_size, rnn_hidden_size, ode_hidden_size, decode_size, cnf_width, teacher_forcing=False,
                 ode_flag=True, cnf_flag=True):
        super().__init__("TrajODEDecoder")

        self.decode_size = decode_size
        self.teacher_forcing = teacher_forcing

        self.ode_flag = ode_flag
        self.cnf_flag = cnf_flag

        self.norm_linear = nn.Linear(encode_size, rnn_hidden_size * 2)

        if cnf_flag:
            cnf_blocks = []
            for _ in range(1):
                cnf_blocks.append(self.construct_cnf(rnn_hidden_size))
            self.cnf_blocks = TrajODEDecoder.SequentialFlow(cnf_blocks)

        self.gru_cell = nn.GRUCell(self.decode_size, rnn_hidden_size)
        self.ode_func = TrajODEDecoder.RecODEFunc(rnn_hidden_size, ode_hidden_size)
        self.st_gate = nn.Sequential(nn.Linear(decode_size, rnn_hidden_size),
                                     nn.ReLU())
        self.z2h = nn.Linear(rnn_hidden_size, rnn_hidden_size)

        self.out_linear = nn.Linear(rnn_hidden_size, decode_size)

        # self.p_z0 = torch.distributions.MultivariateNormal(
        #     loc=torch.zeros(rnn_hidden_size).float(),
        #     covariance_matrix=torch.diag(torch.ones(rnn_hidden_size) * 0.1).float())

    def forward(self, tgt, encode):
        B, L, E = tgt.shape

        mean, logvar = self.norm_linear(encode).chunk(2, -1)
        norm_samples = torch.randn_like(mean)
        z_t0 = torch.exp(logvar * 0.5) * norm_samples + mean

        if self.cnf_flag:
            # z_tk = self.cal_z_tk(z_t0)
            zeros_ = torch.zeros(B, 1).to(z_t0.device)
            z_tk, delta_logp = self.cnf_blocks(z_t0, zeros_)

            # CNF-KL
            log_p_zk = self.log_normal_standard(z_tk, dim=1)
            log_q_z0 = self.log_normal_diag(z_t0, mean=mean, log_var=logvar, dim=1)

            # previous
            sum_logs = torch.sum(log_q_z0 - log_p_zk)
            sum_ldj = torch.sum(-delta_logp.view(-1))
            kl_all = sum_logs - sum_ldj

            kld_loss = kl_all / float(B)

        else:
            z_tk = z_t0
            kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim=1), dim=0)

        t = torch.arange(0, L + 1).float().to(z_t0.device)

        h = self.z2h(z_tk)
        decode_seq = []
        if self.teacher_forcing:
            for i in range(L):
                h_cdd_gru = self.gru_cell(tgt[:, i], h)  # (B, hidden_size)
                h_cdd_ode = odeint(self.ode_func, h, torch.stack([t[i], t[i + 1]]))[-1]

                u = torch.exp(-self.st_gate(tgt[:, i]))
                h = u * h_cdd_ode + (1 - u) * h_cdd_gru
                o = self.out_linear(h)
                decode_seq.append(o)
        else:
            input_ = tgt[:, 0]
            for i in range(L):
                h_cdd_gru = self.gru_cell(input_, h)  # (B, hidden_size)
                h_cdd_ode = odeint(self.ode_func, h, torch.stack([t[i], t[i + 1]]))[-1]
                u = torch.exp(-self.st_gate(tgt[:, i]))
                h = u * h_cdd_ode + (1 - u) * h_cdd_gru
                o = self.out_linear(h)
                decode_seq.append(o)
                input_ = o

        decode_seq = torch.stack(decode_seq, dim=1)

        return decode_seq, kld_loss

    # def cal_ode(self, x):
    #     t0, t1 = 0, 1
    #     logp_diff_t1 = torch.zeros(x.size(0), 1).float().to(x.device)
    #
    #     z_t, logp_diff_t = odeint(
    #         self.cnf,
    #         (x, logp_diff_t1),
    #         torch.tensor([t1, t0]).float().to(x.device),
    #         atol=1e-5,
    #         rtol=1e-5,
    #         method='dopri5'
    #     )
    #
    #     z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]
    #     logp_x = self.p_z0.log_prob(z_t0).to(x.device) - logp_diff_t0.reshape(-1)
    #     p_x = torch.exp(logp_x)
    #     return p_x

    @staticmethod
    def construct_cnf(n_hidden):
        diffeq = TrajODEDecoder.ODECNFnet(n_hidden, n_hidden)
        odefunc = ODECNFfunc(
            diffeq=diffeq,
            divergence_fn="approximate",
            residual=False,
            rademacher=True,
        )
        cnf = CNF(
            odefunc=odefunc,
            T=1.0,
            # train_T=True,
            train_T=False,
            regularization_fns=None,
        )
        return cnf

    class SequentialFlow(nn.Module):
        def __init__(self, layersList):
            super().__init__()
            self.chain = nn.ModuleList(layersList)

        def forward(self, x, logpx=None, reverse=False, inds=None):
            if inds is None:
                if reverse:
                    inds = range(len(self.chain) - 1, -1, -1)
                else:
                    inds = range(len(self.chain))
            if logpx is None:
                for i in inds:
                    x = self.chain[i](x, reverse=reverse)
                return x
            else:
                for i in inds:
                    x, logpx = self.chain[i](x, logpx, reverse=reverse)
                return x, logpx

    class ODECNFnet(nn.Module):
        def __init__(self, inplanes, units=64):
            super().__init__()
            self.relu = nn.ReLU(inplace=True)
            self.fc1 = TrajODEDecoder.ConcatLinear(inplanes, units)
            self.fc2 = TrajODEDecoder.ConcatLinear(units, units)
            self.fc3 = TrajODEDecoder.ConcatLinear(units, units)
            self.nfe = 0

        def forward(self, t, x):
            self.nfe += 1
            out = self.fc1(t, x)
            out = self.relu(out)
            out = self.fc2(t, out)
            out = self.relu(out)
            out = self.fc3(t, out)
            return out

    class ConcatLinear(nn.Module):
        def __init__(self, dim_in, dim_out):
            super().__init__()
            self._layer = nn.Linear(dim_in + 1, dim_out)

        def forward(self, t, x):
            tt = torch.ones_like(x[:, :1]) * t
            ttx = torch.cat([tt, x], 1)
            return self._layer(ttx)

    class RecODEFunc(nn.Module):
        def __init__(self, latent_dim, nhidden):
            super().__init__()
            self.act = nn.Tanh()
            self.nhidden = nhidden
            self.fc1 = nn.Linear(latent_dim, nhidden)
            self.fc2 = nn.Linear(nhidden, nhidden)
            self.fc3 = nn.Linear(nhidden, latent_dim)
            self.nfe = 0
            self.reset_parameters()

        def reset_parameters(self):
            stdv = 1.0 / math.sqrt(self.nhidden)
            for weight in self.parameters():
                nn.init.uniform_(weight, -stdv, stdv)
            return True

        def forward(self, t, x):
            self.nfe += 1
            out = self.fc1(x)
            out = self.act(out)
            out = self.fc2(out)
            out = self.act(out)
            out = self.fc3(out)
            return out

    @staticmethod
    def log_normal_standard(x, average=False, reduce=True, dim=None):
        log_norm = -0.5 * x * x
        if reduce:
            if average:
                return torch.mean(log_norm, dim)
            else:
                return torch.sum(log_norm, dim)
        else:
            return log_norm

    # z0
    @staticmethod
    def log_normal_diag(x, mean, log_var, average=False, reduce=True, dim=None):
        log_norm = -0.5 * (log_var + (x - mean) * (x - mean)
                           * log_var.exp().reciprocal())
        if reduce:
            if average:
                return torch.mean(log_norm, dim)
            else:
                return torch.sum(log_norm, dim)
        else:
            return log_norm
