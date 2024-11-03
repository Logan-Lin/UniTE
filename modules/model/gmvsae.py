import torch
from torch.nn.utils.rnn import pack_padded_sequence
from einops import rearrange, repeat

from modules.model.base import *
from modules.model.rnn import RnnEncoder, RnnDecoder


class LatentGaussMixture(nn.Module):
    def __init__(self, num_cluster, rnn_dim, embed_size):
        super(LatentGaussMixture, self).__init__()
        self.num_cluster = num_cluster
        self.rnn_dim = rnn_dim
        self.embed_size = embed_size

        self.mu_c = nn.Parameter(torch.rand(num_cluster, embed_size))
        self.log_sigma_sq_c = nn.Parameter(
            torch.zeros(num_cluster, embed_size, requires_grad=False))  # source implementation

        self.fc_mu_z = nn.Linear(rnn_dim, embed_size, bias=True)
        self.fc_sigma_z = nn.Linear(rnn_dim, embed_size, bias=True)

    def forward(self, embedded_state, train=False):
        return self.post_estimate(embedded_state, train)

    def post_estimate(self, embedded_state, train=False):
        """
        embedded_state: (batch_size, rnn_dim)
        """
        bs, _ = embedded_state.size()

        mu_z = self.fc_mu_z(embedded_state)
        log_sigma_sq_z = self.fc_sigma_z(embedded_state)

        eps_z = torch.randn_like(log_sigma_sq_z).to(log_sigma_sq_z.device)
        z = mu_z + torch.sqrt(torch.exp(log_sigma_sq_z)) * eps_z  # reparameterization trick

        if train:
            return z, mu_z, log_sigma_sq_z
        else:
            return z


class GMVSAEEncoder(Encoder):
    """
    Variational Autoencoder with Gaussian Mixture Model.
    Liu, Yiding, et al. "Online anomalous trajectory detection with deep generative sequence modeling." 2020 IEEE 36th International Conference on Data Engineering (ICDE). IEEE, 2020.
    """
    def __init__(self, rnn_type, num_embed, d_model, hidden_size, rnn_output_size,
                 num_cluster, output_size, sampler, grid_col=None, aux_cols=None):
        super().__init__(sampler, f'GMVSAE-{rnn_type}-' + str(grid_col) +
                         f'-d{d_model}-h{hidden_size}-o{rnn_output_size}')

        self.output_size = output_size

        self.rnn_encoder = RnnEncoder(rnn_type, num_embed, d_model, hidden_size,
                                      1, rnn_output_size, False, sampler, road_col=grid_col)
        self.latent_gauss_mixture = LatentGaussMixture(num_cluster, rnn_output_size, output_size)

    def forward(self, trip, valid_len, train=False):
        bs, length = trip.size(0), trip.size(1)

        h_final = self.rnn_encoder(trip, valid_len)

        z = self.latent_gauss_mixture(h_final, train)

        return z


class GMVSAEDecoder(Decoder):
    def __init__(self, rnn_type, num_embed, encode_size, d_model, hidden_size,
                 grid_col=None):
        super().__init__(f'GMVSAE-{rnn_type}-' + str(grid_col) +
                         f'-d{d_model}-h{hidden_size}')

        self.rnn_decoder = RnnDecoder(rnn_type, num_embed, encode_size, d_model, hidden_size,
                                      num_layers=1, road_col=grid_col)

    def forward(self, trip, valid_len, z):
        bs, length = trip.size(0), trip.size(1)

        _, _, recon_loss = self.rnn_decoder(trip, valid_len, z)

        return recon_loss
