import os
from random import randint
from abc import abstractmethod

import os
import h5py
from sklearn.utils import shuffle
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch import nn
from tqdm import tqdm
from einops import rearrange, repeat
from torch_scatter import scatter_add
from pytorch_msssim import ssim

from pretrain.func import *
from model.transformer import ContinuousEncoding, get_batch_mask, TransformerDecoder
from data import Denormalizer, TRIP_COLS
from utils import pad_arrays_pair, next_batch, random_subseq, next_batch_recycle


class AutoRegressive(nn.Module):
    def __init__(self, flat_valid, out_dis, out_con_feats, dis_weight, con_weight, latent_size):
        super().__init__()
        self.name = f'AutoReg_fv{int(flat_valid)}_latent{latent_size}' + \
            f'_dis-{dis_weight}-' + ','.join(map(str, out_dis['feats'])) + \
            f'_con-{con_weight}-' + ','.join(map(str, out_con_feats))

        self.flat_valid = flat_valid
        self.out_dis = out_dis
        self.out_con_feats = out_con_feats
        self.dis_weight = dis_weight
        self.con_weight = con_weight

        self.dis_embeds = nn.ModuleList([nn.Sequential(nn.Embedding(num_embed, latent_size),
                                                       nn.LayerNorm(latent_size))
                                        for num_embed in out_dis['num_embeds']])
        self.dis_pres = nn.ModuleList([nn.Sequential(nn.Linear(latent_size, latent_size * 4, bias=False),
                                                     nn.LayerNorm(latent_size * 4),
                                                     nn.LeakyReLU(inplace=True),
                                                     nn.Linear(latent_size * 4, num_embed),
                                                     nn.Softplus())
                                       for num_embed in out_dis['num_embeds']])
        self.con_embed = nn.Linear(len(out_con_feats), latent_size)
        self.con_pre = nn.Sequential(nn.Linear(latent_size, latent_size // 4, bias=False),
                                     nn.LayerNorm(latent_size // 4),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Linear(latent_size // 4, len(out_con_feats)))
        self.start_token = nn.Parameter(torch.randn(latent_size).float(), requires_grad=True)

    def _embed(self, x):
        h = torch.stack([dis_embed(x[..., i].long())
                         for i, dis_embed in enumerate(self.dis_embeds)], -1).sum(-1)
        h = h + self.con_embed(x[..., -len(self.out_con_feats):])
        return h

    def forward(self, models, enc_metas, rec_metas, *args):
        encoder, decoder = models
        # Feed encode metas to the encoder.
        encode = encoder(*enc_metas)

        # Calculate target embedding sequence.
        trip, lengths = rec_metas
        B, L, _ = trip.shape
        tgt_feat = trip[..., self.out_dis['feats'] + self.out_con_feats]  # (B, L, E_feat)
        tgt_latent = torch.cat([repeat(self.start_token, 'E -> B 1 E', B=B),
                                self._embed(tgt_feat[:, :-1])], 1)  # (B, L, E_latent)

        # Calculate recovery loss.
        loss = 0.0
        dec_out = decoder(tgt_latent, encode)  # (B, L, E_latent)
        for i, dis_pre in enumerate(self.dis_pres):
            pred, label = DDPM._flat(self.flat_valid, dis_pre(dec_out),
                                     tgt_feat[..., i].long(), length=lengths)
            loss += F.cross_entropy(pred, label) * self.dis_weight
        if len(self.out_con_feats) > 0:
            pred, label = DDPM._flat(self.flat_valid, self.con_pre(dec_out),
                                     tgt_feat[..., -len(self.out_con_feats):], length=lengths)
            loss += F.mse_loss(pred, label) * self.con_weight
        return loss

    @torch.no_grad()
    def generate(self, models, enc_metas, rec_metas):
        encoder, decoder = models
        # Feed encode metas to the encoder.
        encode = encoder(*enc_metas)

        # Calculate target embedding sequence.
        trip, lengths = rec_metas
        B, L, _ = trip.shape
        cur = repeat(self.start_token, 'E -> B 1 E', B=B)  # (B, 1, E_latent)

        pred_feats = []
        for l in range(L):
            dec_out = decoder(cur, encode)[:, -1]  # (B, 1, E_latent)
            pred_feat = torch.stack([dis_pre(dec_out).argmax(-1) for dis_pre in self.dis_pres], -1)
            pred_feat = torch.cat([pred_feat, self.con_pre(dec_out)], -1)  # (B, E_feat)
            pred_feats.append(pred_feat)
            cur = torch.cat([cur, self._embed(pred_feat.unsqueeze(1))], 1)
        pred_feats = torch.stack(pred_feats, 1)  # (B, L, E_feat)
        result = {f'pred_w_{i}': pred_feats[..., i].long().cpu().numpy() for i in range(len(self.dis_pres))}
        result['pred_con'] = pred_feats[..., -len(self.out_con_feats):].cpu().numpy()
        return result, 'AutoReg'


class MLM(nn.Module):
    """ 
    Masked Language Model. 

    Devlin J, Chang M W, Lee K, et al. Bert: Pre-training of deep bidirectional transformers for language understanding.
    """

    def __init__(self, out_dis, out_con_feats, dis_weight, con_weight, latent_size):
        super().__init__()
        self.name = f'MLM-dis-{dis_weight}-' + ','.join(map(str, out_dis['feats'])) + \
            (f'-con-{con_weight}-' + ','.join(map(str, out_con_feats)) if out_con_feats else '')

        self.dis_feats = out_dis['feats']
        self.dis_num_classes = out_dis['num_embeds']
        self.con_feats = out_con_feats
        self.dis_weight = dis_weight
        self.con_weight = con_weight

        self.dis_pres = nn.ModuleList([nn.Sequential(nn.Linear(latent_size, latent_size * 4, bias=False),
                                                     nn.LayerNorm(latent_size * 4),
                                                     nn.LeakyReLU(inplace=True),
                                                     nn.Linear(latent_size * 4, num_class),
                                                     nn.Softplus())
                                       for num_class in self.dis_num_classes])
        self.con_pre = nn.Sequential(nn.Linear(latent_size, latent_size // 4, bias=False),
                                     nn.LayerNorm(latent_size // 4),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Linear(latent_size // 4, len(out_con_feats)))

    def forward(self, models, common_meta, enc_metas, rec_metas):
        model, = models
        latent = model(*enc_metas, *common_meta, pretrain=True)
        latent = torch.where(torch.isnan(latent), torch.full_like(latent, 1e-4), latent)

        trip, lengths = rec_metas[0], rec_metas[1]
        loss = 0.0
        for i, dis_pre in enumerate(self.dis_pres):
            pred, label = DDPM._flat(True, dis_pre(latent),
                                     trip[..., self.dis_feats[i]].long(), length=lengths)
            loss += F.cross_entropy(pred, label) * self.dis_weight
        if len(self.con_feats) > 0:
            pred, label = DDPM._flat(True, self.con_pre(latent),
                                     trip[..., self.con_feats], length=lengths)
            loss += F.mse_loss(pred, label) * self.con_weight
        return loss


class DDPM(nn.Module):
    """
    Generative loss based on the improved diffusion model proposed in Diffusion-LM.
    X. L. Li, J. Thickstun, I. Gulrajani, P. Liang, and T. B. Hashimoto, “Diffusion-LM Improves Controllable Text Generation.”
    https://github.com/XiangLi1999/Diffusion-LM
    """

    def __init__(self, T, noise_schedule_name, denoise_type, supervise_types, flat_valid,
                 in_dis, out_dis, con_feats):
        super().__init__()
        self.name = f'DDPM_T{T}_{noise_schedule_name}_{denoise_type}_SV' + \
            ','.join(supervise_types) + f'_fv{int(flat_valid)}' + \
            '_in' + ','.join(map(str, in_dis['dis_feats'])) + '-' + ','.join(map(str, con_feats)) + \
            '_out' + ';'.join([f'{arg_i},{dis_feat},{dis_pre_i}'
                               for arg_i, dis_feat, dis_pre_i
                               in zip(out_dis['feat_arg_i'], out_dis['dis_feats'], out_dis['dis_pre_i'])])

        self.T = T
        self.denoise_type = denoise_type
        self.noise_schedule_name = noise_schedule_name

        self.in_dis = in_dis
        self.out_dis = out_dis
        self.con_feats = con_feats
        self.supervise_types = supervise_types
        self.flat_valid = flat_valid

        if len(in_dis['dis_feats']):
            assert len(in_dis['dis_feats']) == len(in_dis['num_dis_embeds']), \
                'length of num_embeds list should be equal to the number of discrete features.'
            self.dis_embeds = nn.ModuleList([nn.Sequential(nn.Embedding(num_embed, in_dis['dis_embed_size']),
                                                           nn.LayerNorm(in_dis['dis_embed_size']))
                                             for num_embed in in_dis['num_dis_embeds']])
        else:
            self.dis_embeds = None

        self.x0_dis_size = len(in_dis['dis_feats']) * in_dis['dis_embed_size']
        if len(out_dis['num_dis_embeds']):
            self.dis_predictors = nn.ModuleList([nn.Sequential(nn.Linear(self.x0_dis_size, self.x0_dis_size * 4, bias=False),
                                                               nn.LayerNorm(self.x0_dis_size * 4),
                                                               nn.LeakyReLU(inplace=True),
                                                               nn.Linear(self.x0_dis_size * 4, num_embed),
                                                               nn.Softplus())
                                                 for num_embed in out_dis['num_dis_embeds']])
        else:
            self.dis_predictors = None

        # define beta schedule
        if noise_schedule_name == 'cosine':
            self.schedule_func = cosine_beta_schedule
        elif noise_schedule_name == 'quadratic':
            self.schedule_func = quadratic_beta_schedule
        elif noise_schedule_name == 'sigmoid':
            self.schedule_func = sigmoid_beta_schedule
        elif noise_schedule_name == 'linear':
            self.schedule_func = linear_beta_schedule
        else:
            raise NotImplementedError(noise_schedule_name)

        self.betas = self.schedule_func(T)
        alphas = 1. - self.betas
        self.alphas = alphas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.alphas_cumprod = alphas_cumprod

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    def cal_true_xstart(self, trip):
        """
        Calculates the true x0 given the trip sequence.

        :returns: the true x0, with shape (B, L, num_con_feats + num_dis_feats * dis_embed_size)
        """
        rec_target = []
        if len(self.con_feats):
            rec_target.append(trip[..., self.con_feats])
        if self.dis_embeds is not None:
            rec_target += [dis_embed(trip[..., dis_feat].long())
                           for dis_feat, dis_embed in
                           zip(self.in_dis['dis_feats'], self.dis_embeds)]
        rec_target = torch.cat(rec_target, -1)
        return rec_target

    def pred_w(self, x_0, ws, train=True):
        """
        Predict discrete tokens based on the continuous denoised feature x_0.
        """
        pred, target, lengths = {}, {}, {}
        x0_dis_embeds = x_0[..., -self.x0_dis_size:]  # (B, L, x0_dis_size)
        for i, (feat_arg_i, cand_arg_i, len_arg_i, dis_feat, seq_i_feat, dis_pre_i) \
            in enumerate(zip(self.out_dis['feat_arg_i'], self.out_dis['cand_arg_i'],
                             self.out_dis['len_arg_i'], self.out_dis['dis_feats'],
                             self.out_dis['seq_i_feats'], self.out_dis['dis_pre_i'])):
            w_arg = ws[feat_arg_i]
            len_arg = ws[len_arg_i].long()
            dis_true = w_arg[..., dis_feat].long()
            dis_seq_i = w_arg[..., seq_i_feat].long()  # (B, L)

            B, L, _ = x0_dis_embeds.shape
            resampled_L = dis_seq_i.size(1)
            dis_seq_i_flat = (dis_seq_i +
                              torch.arange(B).to(dis_seq_i.device).unsqueeze(1) * L).reshape(-1)

            embed = x0_dis_embeds.reshape(B * L, -1)[dis_seq_i_flat].reshape(B, resampled_L, -1)
            dis_predictor = self.dis_predictors[dis_pre_i]
            dis_pre = dis_predictor(embed)
            if cand_arg_i != -1:
                num_class = dis_pre.size(-1)
                cand = ws[cand_arg_i].long()
                cand = cand.masked_fill_(cand < 0, num_class)  # (B, L, N_cand)
                cand_mask = scatter_add(torch.ones_like(cand), cand, dim=-1,
                                        dim_size=num_class + 1)[..., :num_class].bool()
                cand_mask = cand_mask.reshape(B * L, -1)[dis_seq_i_flat].reshape(B, resampled_L, -1)
                dis_pre = dis_pre.masked_fill(~cand_mask, 0.0)
            if train:
                dis_pre, dis_true = self._flat(self.flat_valid, dis_pre, dis_true, length=len_arg)
            else:
                dis_pre = dis_pre.argmax(-1)
            pred[f'pred_w_{i}'] = dis_pre
            target[f'target_w_{i}'] = dis_true
            lengths[f'length_w_{i}'] = len_arg
        return pred, target, lengths

    @staticmethod
    def _flat(flat_valid, *xs, length):
        if flat_valid:
            return (pack_padded_sequence(x, length.long().cpu(), batch_first=True, enforce_sorted=False).data
                    for x in xs)
        else:
            return (x.reshape(-1, x.size(-1)) if len(x.shape) > 2 else x.reshape(-1) for x in xs)

    def forward(self, models, enc_metas, rec_metas, *args):
        trips, trip_lengths, *ws = rec_metas
        true_xstart = self.cal_true_xstart(trips)  # (B, L, E)
        noise = torch.randn_like(true_xstart)
        t = sample_t(self.T, true_xstart.size(0), true_xstart.device)
        x_t = self._q_sample(true_xstart, t, noise)

        # Encode trips into conditional information.
        encoder, denoiser = models
        encode = encoder(*enc_metas)

        # Calculcates the predicted noise and x0 based on the output of denoiser.
        model_out = denoiser(x_t, t, encode, trip_lengths)
        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, true_xstart.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, true_xstart.shape)
        if self.denoise_type == 'noise':
            pred_xstart = (x_t - model_out * sqrt_one_minus_alpha_cumprod_t) / sqrt_alpha_cumprod_t
            pred_noise = model_out
        elif self.denoise_type == 'start':
            pred_xstart = model_out
            pred_noise = (x_t - sqrt_alpha_cumprod_t * model_out) / sqrt_one_minus_alpha_cumprod_t

        loss = 0.0
        for supervise_type in self.supervise_types:
            sv_on, sv_func, sv_weight = supervise_type.split('-')
            sv_weight = float(sv_weight)

            if sv_on == 'xstart':
                target, pred = self._flat(self.flat_valid, true_xstart, pred_xstart, length=trip_lengths)
            elif sv_on == 'noise':
                target, pred = self._flat(self.flat_valid, noise, pred_noise, length=trip_lengths)
            elif sv_on in ['truedis', 'preddis']:
                if sv_on == 'truedis':
                    x_0 = true_xstart
                else:
                    x_0 = pred_xstart
                pred, target, _ = self.pred_w(x_0, ws)
                pred = list(pred.values())
                target = list(target.values())
            elif sv_on == 'predcon':
                pred = pred_xstart[..., :len(self.con_feats)]
                target = true_xstart[..., :len(self.con_feats)]
                pred, target = self._flat(self.flat_valid, pred, target, length=trip_lengths)
            else:
                raise NotImplementedError(sv_on)

            if sv_func == 'l1':
                sv_loss = F.l1_loss(pred, target)
            elif sv_func == 'l2':
                sv_loss = F.mse_loss(pred, target)
            elif sv_func == 'huber':
                sv_loss = F.smooth_l1_loss(pred, target)
            elif sv_func == 'ent':
                sv_loss = torch.stack([F.cross_entropy(p, t) for p, t in zip(pred, target)]).sum()
            else:
                raise NotImplementedError(sv_func)

            loss += sv_loss * sv_weight

        return loss

    @torch.no_grad()
    def generate(self, models, enc_metas, rec_metas, **param):
        trips, trip_lengths, *ws = rec_metas
        encoder, denoiser = models
        encode = encoder(*enc_metas)
        true_xstart = self.cal_true_xstart(trips)

        sample_type = param.get('sample_type', 'ddim')
        if sample_type == 'ddpm':
            gen_name = 'DDPM'
            gen_xstart = self._p_sample_loop(denoiser, true_xstart.shape, encode)
        elif sample_type == 'ddim':
            step_schedule_name = param['step_schedule_name']
            DDIM_T = param['DDIM_T']
            gen_name = f'DDIM{DDIM_T}_{step_schedule_name}_eta{param["eta"]}'

            if step_schedule_name == 'linear':
                skip = self.T // DDIM_T
                step_seq = range(0, self.T, skip)
            elif step_schedule_name == 'quad':
                step_seq = [int(t) for t in (np.linspace(0, np.sqrt(self.T * 0.8), DDIM_T) ** 2)]
            else:
                raise NotImplementedError(step_schedule_name)
            gen_xstart = self._ddim_sample_loop(denoiser, true_xstart.shape,
                                                encode, step_seq=step_seq, **param)
        else:
            raise NotImplementedError(sample_type)

        results = {}
        # Predict generated discrete tokens.
        if len(self.out_dis['dis_feats']):
            pred_w, _, _ = self.pred_w(gen_xstart, ws, train=False)
            results |= pred_w
        # Fetch generated continuous features.
        if len(self.con_feats):
            results['pred_con'] = gen_xstart[..., :len(self.con_feats)]
        results = {key: value.cpu().numpy() for key, value in results.items()}

        return results, gen_name

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def _q_sample(self, x_start, t, noise=None):
        """ Calculate the noisy data q(x_t|x_0). """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def _compute_alpha_cumprod(self, t, num_dim):
        beta = torch.cat([torch.zeros(1), self.betas], dim=0).to(t.device)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, *([1] * (num_dim - 1)))
        return a

    def _p_sample_loop(self, model, shape, y):
        """ Finish the DDPM reversed denoising diffusion process. """
        device = next(model.parameters()).device
        x = torch.randn(*shape, device=device)
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=device).long()
            model_out = model(x, t, y)
            noise = torch.randn_like(x)
            alpha_cumprod_t = self._compute_alpha_cumprod(t, len(x.shape))
            sqrt_one_minus_alphas_cumprod_t = (1 - alpha_cumprod_t).sqrt()

            alpha_cumprod_tnext = self._compute_alpha_cumprod(t - 1, len(x.shape))
            sqrt_alpha_cumprod_tnext = alpha_cumprod_tnext.sqrt()
            sqrt_one_minus_alpha_cumprod_tnext = (1 - alpha_cumprod_tnext).sqrt()
            if self.denoise_type == 'noise':
                betas_t = self._extract(self.betas, t, x.shape)
                sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)
                mean = sqrt_recip_alphas_t * (x - betas_t * model_out / sqrt_one_minus_alphas_cumprod_t)
                posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
                if i == 0:
                    sample = mean
                else:
                    sample = mean + torch.sqrt(posterior_variance_t) * noise
            elif self.denoise_type == 'start':
                if i == 0:
                    sample = model_out
                else:
                    sample = sqrt_alpha_cumprod_tnext * model_out + \
                        sqrt_one_minus_alpha_cumprod_tnext * noise
            x = sample
        return x

    def _ddim_sample_loop(self, model, shape, y, step_seq, **gen_param):
        """ Finish the DDIM accelerated reversed denoising diffusion process. """
        b = shape[0]
        seq_next = [-1] + list(step_seq[:-1])
        device = next(model.parameters()).device
        x = torch.randn(*shape, device=device)
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(step_seq), reversed(seq_next)):
            t, next_t = (torch.ones(b) * i).to(x.device).long(), (torch.ones(b) * j).to(x.device).long()
            alpha_cumprod_t = self._compute_alpha_cumprod(t, len(x.shape))
            alpha_cumprod_next = self._compute_alpha_cumprod(next_t, len(x.shape))
            sqrt_alpha_cumprod_t = alpha_cumprod_t.sqrt()
            sqrt_one_minus_alpha_cumprod_t = (1 - alpha_cumprod_t).sqrt()

            xt = xs[-1].to(x.device)
            if self.denoise_type == 'noise':
                epsilon = model(xt, t, y=y)
                x0_pred = (xt - epsilon * sqrt_one_minus_alpha_cumprod_t) / sqrt_alpha_cumprod_t
            elif self.denoise_type == 'start':
                x0_pred = model(xt, t, y=y)
                epsilon = (xt - sqrt_alpha_cumprod_t * x0_pred) / sqrt_one_minus_alpha_cumprod_t
            x0_preds.append(x0_pred)
            c1 = (gen_param['eta'] * ((1 - alpha_cumprod_t / alpha_cumprod_next) *
                                      (1 - alpha_cumprod_next) / (1 - alpha_cumprod_t)).sqrt())
            c2 = ((1 - alpha_cumprod_next) - c1 ** 2).sqrt()
            xt_next = alpha_cumprod_next.sqrt() * x0_pred + c1 * torch.randn_like(x) + c2 * epsilon
            xs.append(xt_next)

        return xs[-1]


class GMVSAE(nn.Module):
    def __init__(self, num_cluster, rnn_dim, embed_size, pretrain):
        super(GMVSAE, self).__init__()

        self.name = f'GM-VSAE-pre{int(pretrain)}'

        self.num_cluster = num_cluster
        self.rnn_dim = rnn_dim
        self.embed_size = embed_size
        self.pretrain = pretrain

    def forward(self, models, enc_metas, rec_metas, *args):
        assert len(models[:-1]) == 1, "number of encoders must be 1"

        encoder = models[0]
        z, mu_z, log_sigma_sq_z = encoder(*enc_metas, train=True)

        trip, valid_len = rec_metas
        bs = trip.size(0)
        recon_loss = models[-1](trip, valid_len, z)

        if self.pretrain:
            return recon_loss

        stack_z = repeat(z, 'b d -> b c d', c=self.num_cluster)
        stack_mu_c = repeat(encoder.latent_gauss_mixture.mu_c, 'c d -> b c d', b=bs)
        stack_mu_z = repeat(mu_z, 'b d -> b c d', c=self.num_cluster)
        stack_log_sigma_sq_c = repeat(encoder.latent_gauss_mixture.log_sigma_sq_c, 'c d -> b c d', b=bs)
        stack_log_sigma_sq_z = repeat(log_sigma_sq_z, 'b d -> b c d', c=self.num_cluster)

        pi_post_logits = torch.sum(torch.square(stack_z - stack_mu_c) / torch.exp(stack_log_sigma_sq_c), dim=-1)
        pi_post = torch.softmax(pi_post_logits, dim=-1) + 1e-10

        batch_gaussian_loss = 0.5 * torch.sum(
            pi_post *
            torch.mean(
                stack_log_sigma_sq_c +
                torch.exp(stack_log_sigma_sq_z) / torch.exp(stack_log_sigma_sq_c) +
                torch.square(stack_mu_z - stack_mu_c) / torch.exp(stack_log_sigma_sq_c),
                dim=-1
            ),
            dim=-1
        ) - 0.5 * torch.mean(1 + log_sigma_sq_z, dim=-1)
        batch_uniform_loss = torch.mean(torch.mean(pi_post, dim=0) * torch.log(torch.mean(pi_post, dim=0)))

        return recon_loss + 1.0 / self.rnn_dim * torch.mean(batch_gaussian_loss) + 0.1 * torch.mean(batch_uniform_loss)

    # TODO: Generate mu_c by K means
    # def generate_mu_c(self, trip, valid_len, models, sample_num=10000):


class TrajectorySim(nn.Module):
    def __init__(self, device, hidden_size, vocab_size, knn_vocabs_path,
                 criterion_name, dist_decay_speed, timeWeight,
                 use_discriminative, discriminative_w, dis_freq,
                 generator_batch):
        super(TrajectorySim, self).__init__()
        self.name = f'TrajectorySim_vocab{vocab_size}_criterion-{criterion_name}_discriminative{int(use_discriminative)}'
        self.device = device
        self.PAD, self.BOS, self.EOS, self.UNK = 0, 1, 2, 3
        self.vocab_size = vocab_size
        self.knn_vocabs_path = knn_vocabs_path
        self.criterion_name = criterion_name
        self.dist_decay_speed = dist_decay_speed
        self.timeWeight = timeWeight
        self.use_discriminative = use_discriminative
        self.discriminative_w = discriminative_w
        self.dis_freq = dis_freq
        self.generator_batch = generator_batch

        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)  # 三角损失
        self.loss_function = self.set_loss()  # 损失函数

        #  Encoder到Decoder 的中间输出输出到词汇表向量的映射，并进行了log操作
        self.mapping = nn.Sequential(nn.Linear(hidden_size, self.vocab_size), nn.LogSoftmax(dim=1))

    def set_loss(self):
        """
        设置KL散度损失函数

        :param args:  参数设定
        :return: 设置的损失函数
        """
        if self.criterion_name == "NLL":
            criterion = self.nll_criterion().to(self.device)
            return lambda o, t: criterion(o, t)

        else:  # args.criterion_name == KLDIV
            assert os.path.isfile(self.knn_vocabs_path), "{} does not exist".format(self.knn_vocabs_path)
            print("Loading vocab distance file {}...".format(self.knn_vocabs_path))
            with h5py.File(self.knn_vocabs_path, 'r') as f:
                # VD size = (vocal_size, 10) 第i行为第i个轨迹与其10个邻居
                V, Ds, Dt = f["V"][...], f["Ds"][...], f["Dt"][...]
                V, Ds, Dt = torch.LongTensor(V), torch.FloatTensor(Ds), torch.FloatTensor(Dt)
                Ds, Dt = self.dist2weight(Ds, self.dist_decay_speed), self.dist2weight(Dt, self.dist_decay_speed)
                D = (1 - self.timeWeight) * Ds + self.timeWeight * Dt
            V, D = V.to(self.device), D.to(self.device)
            criterion = nn.KLDivLoss(reduction='sum').to(self.device)
            return lambda o, t: self.kl_criterion(o, t, criterion, V, D)

    def set_disLoss_dataGenerator(self, batch_iter, batch_size):
        self.dataGenerator = next_batch_recycle(shuffle(batch_iter), batch_size)
        # self.dataGenerator = next_batch(shuffle(batch_iter), batch_size)

    def EncoderDecoder(self, models, enc_metas, trg):
        """
        Input:
            models: TrajectorySim or t2vec
            enc_metas: encode meta, (src, lengths, _)
                src (batch, src_seq_len): source tensor
                lengths (batch): source sequence lengths
            trg (batch, trg_seq_len): target tensor, the `seq_len` in trg is not
                necessarily the same as that in src
        ---
        Output:
            output (trg_seq_len, batch, hidden_size)
        """
        encoder, decoder, embedding = models
        encoder_hn, H = encoder(*enc_metas, embed_layer=embedding)
        decoder_h0 = encoder.encoder_hn2decoder_h0(encoder_hn)
        # target去除EOS行后调入decoder
        output, decoder_hn = decoder(trg[:, :-1], decoder_h0, H, embed_layer=embedding)
        return output

    def genLoss(self, enc_metas, target, models):
        """
        计算一批训练数据的损失

        :param enc_metas: encode meta
        :param target: (batch, seq_len2)
        :param models: encoder-decoder
        :return:
        """
        output = self.EncoderDecoder(models, enc_metas, target) #  (seq_len2, batch, hidden_size)

        batch = output.size(1)
        loss = 0
        #  we want to decode target in range [BOS+1:EOS]
        target = target.transpose(0, 1)[1:]
        # generator_batch 32每一次生成的words数目，要求内存
        # output [max_target_size, 128, 256]
        for o, t in zip(output.split(self.generator_batch),
                        target.split(self.generator_batch)):
            # (generator_batch, batch, hidden_size) => (batch*generator_batch, hidden_size)
            o = o.view(-1, o.size(2))
            o = self.mapping(o)
            #  (batch*generator_batch,)
            t = t.reshape(-1)
            loss += self.loss_function(o, t)
        return loss.div(batch)

    def disLoss(self, a, p, n, models):
        """
        计算相似性损失，即三角损失

        通过a,p,n三组轨迹，经过前向encoder,接着通过encoder_hn2decoder_h0，取最后一层向量作为每组每个轨迹的代表

        a (named tuple): anchor data
        p (named tuple): positive data
        n (named tuple): negative data
        """
        # a_src (seq_len, 128)
        a_src, a_lengths, a_invp = a.src.to(self.device), a.lengths.to(self.device), a.invp.to(self.device)
        p_src, p_lengths, p_invp = p.src.to(self.device), p.lengths.to(self.device), p.invp.to(self.device)
        n_src, n_lengths, n_invp = n.src.to(self.device), n.lengths.to(self.device), n.invp.to(self.device)

        encoder, _, embedding = models
        # (num_layers * num_directions, batch, hidden_size)  (2*3, 128, 256/2)
        a_h, _ = encoder(a_src, a_lengths, embed_layer=embedding)
        p_h, _ = encoder(p_src, p_lengths, embed_layer=embedding)
        n_h, _ = encoder(n_src, n_lengths, embed_layer=embedding)
        # (num_layers, batch, hidden_size * num_directions) (3,128,256)
        a_h = encoder.encoder_hn2decoder_h0(a_h)
        p_h = encoder.encoder_hn2decoder_h0(p_h)
        n_h = encoder.encoder_hn2decoder_h0(n_h)
        # take the last layer as representations (batch, hidden_size * num_directions) (128,256)
        a_h, p_h, n_h = a_h[-1], p_h[-1], n_h[-1]
        return self.triplet_loss(a_h[a_invp], p_h[p_invp], n_h[n_invp])  # (128,256)


    def forward(self, models, enc_metas, rec_metas, current_iteration_id, *args):
        """
        Input:
            models: TrajectorySim
            enc_metas: encode meta
            rec_metas: recovery meta
            current_iteration_id: current iteration index in pretrain
        ---
        Output:
            train_loss
        """
        _, _, trg = rec_metas
        # 计算生成损失+三元判别损失
        train_gen_loss = self.genLoss(enc_metas, trg, models)
        train_dis_cross, train_dis_inner = torch.tensor(0), torch.tensor(0)
        if self.use_discriminative and current_iteration_id % self.dis_freq == 0:
            # a和p的轨迹更接近 a.src.size = [max_length,128]
            a, p, n = self.get_apn_cross()
            train_dis_cross = self.disLoss(a, p, n, models)
            # a,p,n是由同一组128个轨迹采样得到的新的128个下采样轨迹集合
            a, p, n = self.get_apn_inner()
            train_dis_inner = self.disLoss(a, p, n, models)
        # 损失按一定权重相加 train_gen_loss： 使损失尽可能小 discriminative——loss: 使序列尽可能相似
        # 计算词的平均损失
        train_gen_loss = train_gen_loss / trg.size(1)
        train_dis_loss = train_dis_cross + train_dis_inner
        train_loss = (1 - self.discriminative_w) * train_gen_loss + self.discriminative_w * train_dis_loss
        return train_loss

    def nll_criterion(self):
        """
        带权的NLLLoss损失函数， 将编码为0的填充位置的权重置为0
        :return: 带权损失函数
        """
        weight = torch.ones(self.vocab_size)
        weight[self.PAD] = 0
        # sum-loss求和 null-loss取平均值 none显示全部loss
        criterion = nn.NLLLoss(weight, reduction='sum')
        return criterion

    def kl_criterion(self, output, target, criterion, V, D):
        """
        output (batch, vocab_size)  128*18866
        target (batch,)  128*1
        criterion (nn.KLDIVLoss)
        V (vocab_size, k) 18866*10
        D (vocab_size, k) 18866*10

        该评价模型评价每一批128个目标cell的10个邻居对应的输出权重与真实权重的距离 128*10

        只考虑每个点的10个邻居
        """
        # 获取128个目标cell的10个邻居
        # 第一个参数是索引的对象，第二个参数0表示按行索引，1表示按列进行索引，第三个参数是一个tensor，就是索引的序号
        indices = torch.index_select(V, 0, target)
        # 收集输出的128个目标对应的10个邻居的权重，是模型预测出来的权重
        outputk = torch.gather(output, 1, indices)
        # 获取128个目标cell的10个邻居对应的权重，从D中获取，是真实权重
        targetk = torch.index_select(D, 0, target)
        return criterion(outputk, targetk)

    def dist2weight(self, D, dist_decay_speed=0.8):
        """
        对于K个邻居，按照距离大小给出权重，公式5中的W

        :param D: 和topK邻居的距离矩阵
        :param dist_decay_speed: 衰减指数
        :return:
        """
        # D = D.div(100)
        D = torch.exp(-D * dist_decay_speed)
        s = D.sum(dim=1, keepdim=True)
        D = D / s
        # The PAD should not contribute to the decoding loss
        D[self.PAD, :] = 0.0
        return D

    def get_apn_cross(self):
        """
        得到三个batch个数的轨迹集，a,p,n
        a中的轨迹中心更接近于p中的轨迹
        :return: 选取的一组a, p, n， 每个均为一个TF对象 ['src', 'lengths', 'trg', 'invp']
        """
        def distance(x, y):
            return np.linalg.norm(x[0:2]-y[0:2])
            # return 0.5*np.linalg.norm(x[:2] - y[:2])+ (x[2:3] - y[2:3])*0.5/24

        a_src, a_trg, a_mta = [list(i) for i in zip(*next(self.dataGenerator))]
        p_src, p_trg, p_mta = [list(i) for i in zip(*next(self.dataGenerator))]
        n_src, n_trg, n_mta = [list(i) for i in zip(*next(self.dataGenerator))]

        for i in range(len(a_src)):
            # a_mta[i] float32[] [id, t]
            # 如果a,p两个轨迹距离更大，则将p中的轨迹换为n的轨迹
            if distance(a_mta[i], p_mta[i]) > distance(a_mta[i], n_mta[i]):
                p_src[i], n_src[i] = n_src[i], p_src[i]
                p_trg[i], n_trg[i] = n_trg[i], p_trg[i]
                p_mta[i], n_mta[i] = n_mta[i], p_mta[i]

        a = pad_arrays_pair(a_src, a_trg, keep_invp=True)
        p = pad_arrays_pair(p_src, p_trg, keep_invp=True)
        n = pad_arrays_pair(n_src, n_trg, keep_invp=True)
        return a, p, n

    def get_apn_inner(self):
        """
        以一定概率去除一批batch个数轨迹中的点后生成三个轨迹集合a, p，n

        Test Case:
        a, p, n = dataloader.getbatch_discriminative_inner()
        i = 2
        idx_a = torch.nonzero(a[2].t()[a[3]][i])
        idx_p = torch.nonzero(p[2].t()[p[3]][i])
        idx_n = torch.nonzero(n[2].t()[n[3]][i])
        a_t = a[2].t()[a[3]][i][idx_a].view(-1).numpy()
        p_t = p[2].t()[p[3]][i][idx_p].view(-1).numpy()
        n_t = n[2].t()[n[3]][i][idx_n].view(-1).numpy()
        print(len(np.intersect1d(a_t, p_t)))
        print(len(np.intersect1d(a_t, n_t)))
        """
        a_src, a_trg = [], []
        p_src, p_trg = [], []
        n_src, n_trg = [], []

        _, trgs, _ = [list(i) for i in zip(*next(self.dataGenerator))]
        for i in range(len(trgs)):
            trg = trgs[i][1:-1]
            if len(trg) < 10: continue
            a1, a3, a5 = 0, len(trg)//2, len(trg)
            a2, a4 = (a1 + a3)//2, (a3 + a5)//2
            rate = np.random.choice([0.5, 0.6, 0.8])
            if np.random.rand() > 0.5:
                a_src.append(random_subseq(trg[a1:a4], rate))
                a_trg.append(np.r_[self.BOS, trg[a1:a4], self.EOS])
                p_src.append(random_subseq(trg[a2:a5], rate))
                p_trg.append(np.r_[self.BOS, trg[a2:a5], self.EOS])
                n_src.append(random_subseq(trg[a3:a5], rate))
                n_trg.append(np.r_[self.BOS, trg[a3:a5], self.EOS])
            else:
                a_src.append(random_subseq(trg[a2:a5], rate))
                a_trg.append(np.r_[self.BOS, trg[a2:a5], self.EOS])
                p_src.append(random_subseq(trg[a1:a4], rate))
                p_trg.append(np.r_[self.BOS, trg[a1:a4], self.EOS])
                n_src.append(random_subseq(trg[a1:a3], rate))
                n_trg.append(np.r_[self.BOS, trg[a1:a3], self.EOS])
        a = pad_arrays_pair(a_src, a_trg, keep_invp=True)
        p = pad_arrays_pair(p_src, p_trg, keep_invp=True)
        n = pad_arrays_pair(n_src, n_trg, keep_invp=True)
        return a, p, n


class Trajectory2Vec(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'trajectory2vec_loss'
        
    def forward(self, models, enc_metas, rec_metas):
        loss_func = nn.MSELoss()
        trip, valid_len = enc_metas
        embedding = models[0](trip, valid_len)
        x, c_out = models[1](trip, valid_len, embedding)
        loss = loss_func(c_out, x)
        return loss


class t2vec(TrajectorySim):
    def __init__(self, device, hidden_size, vocab_size, knn_vocabs_path,
                 criterion_name, dist_decay_speed,
                 use_discriminative, discriminative_w,
                 generator_batch):
        super().__init__(device, hidden_size, vocab_size, knn_vocabs_path,
                 criterion_name, dist_decay_speed, timeWeight=0,
                 use_discriminative=use_discriminative, discriminative_w=discriminative_w, dis_freq=10,
                 generator_batch=generator_batch)
        # 重写
        self.name = f't2vec_vocab{vocab_size}_criterion-{criterion_name}_discriminative{int(use_discriminative)}'
        '''
        self.loss_function = self.set_loss()  # 会使用重写的self.set_loss()
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        # Encoder到Decoder的中间输出输出到词汇表向量的映射
        self.mapping = nn.Sequential(nn.Linear(hidden_size, self.vocab_size),
                                     nn.LogSoftmax(dim=1)) 

        self.device = device
        self.PAD, self.BOS, self.EOS, self.UNK = 0, 1, 2, 3
        self.vocab_size = vocab_size
        self.knn_vocabs_path = knn_vocabs_path
        self.criterion_name = criterion_name
        self.dist_decay_speed = dist_decay_speed
        self.timeWeight = 0   # 弃用
        self.use_discriminative = use_discriminative
        self.discriminative_w = discriminative_w
        self.dis_freq = 10
        self.generator_batch = generator_batch
        '''

    def set_loss(self):
        """
        设置KL散度损失函数

        :param args:  参数设定
        :return: 设置的损失函数
        """
        if self.criterion_name == "NLL":
            criterion = self.nll_criterion().to(self.device)
            return lambda o, t: criterion(o, t)

        else:  # args.criterion_name == KLDIV
            assert os.path.isfile(self.knn_vocabs_path), "{} does not exist".format(self.knn_vocabs_path)
            print("Loading vocab distance file {}...".format(self.knn_vocabs_path))
            with h5py.File(self.knn_vocabs_path, 'r') as f:
                # VD size = (vocal_size, K) 第i行为第i个轨迹与其K个邻居
                V, D = f["V"][...], f["D"][...]
                V, D = torch.LongTensor(V), torch.FloatTensor(D)
                D = self.dist2weight(D, self.dist_decay_speed)
            V, D = V.to(self.device), D.to(self.device)
            criterion = nn.KLDivLoss(reduction='sum').to(self.device)
            return lambda o, t: self.kl_criterion(o, t, criterion, V, D)

    def forward(self, models, enc_metas, rec_metas, current_iteration_id, *args):
        """
        Input:
            models: t2vec
            enc_metas: encode meta
            rec_metas: recovery meta
            current_iteration_id: current iteration index in pretrain
        ---
        Output:
            train_loss
        """
        _, _, trg = rec_metas
        # 计算生成损失+三元判别损失
        train_gen_loss = self.genLoss(enc_metas, trg, models)
        train_dis_cross, train_dis_inner = torch.tensor(0), torch.tensor(0)
        if self.use_discriminative and current_iteration_id % self.dis_freq == 0:
            # a和p的轨迹更接近 a.src.size = [max_length,128]
            a, p, n = self.get_apn_cross()
            train_dis_cross = self.disLoss(a, p, n, models)
            # a,p,n是由同一组128个轨迹采样得到的新的128个下采样轨迹集合
            a, p, n = self.get_apn_inner()
            train_dis_inner = self.disLoss(a, p, n, models)
        # 损失按一定权重相加 train_gen_loss： 使损失尽可能小 discriminative——loss: 使序列尽可能相似
        train_loss = train_gen_loss + self.discriminative_w * (train_dis_cross + train_dis_inner)
        return train_loss

    def dist2weight(self, D, dist_decay_speed=0.8):
        """
        对于K个邻居，按照距离大小给出权重，公式5中的W

        :param D: 和topK邻居的距离矩阵
        :param dist_decay_speed: 衰减指数
        :return:
        """
        D = D.div(100)
        D = torch.exp(-D * dist_decay_speed)
        s = D.sum(dim=1, keepdim=True)
        D = D / s
        # The PAD should not contribute to the decoding loss
        D[self.PAD, :] = 0.0
        return D


class Trembr(nn.Module):
    def __init__(self, flat_valid, num_roads, dis_weight, latent_size):
        super().__init__()
        self.name = f'Trembr_fv{int(flat_valid)}_latent{latent_size}'

        self.flat_valid = flat_valid
        self.beta_ = dis_weight

        self.road_embed = nn.Embedding(num_roads, latent_size-1)
        self.road_pre = nn.Sequential(nn.Linear(latent_size, num_roads, bias=False),
                                      nn.Softplus())
        self.time_pre = nn.Sequential(nn.Linear(latent_size, latent_size // 2),
                                      nn.Linear(latent_size // 2, 1))

        self.start_token = nn.Parameter(torch.randn(latent_size).float(), requires_grad=True)

    def _embed(self, x, time_diff):
        h = self.road_embed(x.long())
        h = torch.cat([h, time_diff.unsqueeze(2)], dim=2)
        return h

    def forward(self, models, enc_metas, rec_metas, *args):
        encoder, decoder = models
        # Feed encode metas to the encoder.
        encode = encoder(*enc_metas)

        # Calculate target embedding sequence.
        trip, lengths, time_diff = rec_metas
        B, L = trip.shape
        tgt_latent = torch.cat([repeat(self.start_token, 'E -> B 1 E', B=B),
                                self._embed(trip, time_diff)], 1)  # (B, L, E_latent)

        # Calculate recovery loss.
        loss = 0.0
        dec_out = decoder(tgt_latent, encode)  # (B, L, E_latent)
        pred, label = DDPM._flat(self.flat_valid, self.road_pre(dec_out),
                                 trip.long(), length=lengths)
        loss += self.beta_ * F.cross_entropy(pred, label)

        pred, label = DDPM._flat(self.flat_valid, self.time_pre(dec_out),
                                 time_diff, length=lengths)
        loss += (1 - self.beta_) * torch.log(abs(pred.squeeze() - label) + 1e-4).mean()
        return loss


class ConvolutionalAutoRegressive(nn.Module):
    def __init__(self, recovery_weight):
        super().__init__()
        self.name = f'CAE-{recovery_weight}'

        self.lambda_ = recovery_weight

    def forward(self, models, enc_metas, rec_metas, *args):
        encoder, decoder = models
        # Feed encode metas to the encoder.
        encode = encoder(*enc_metas)

        # Calculate target embedding sequence.
        tgt = rec_metas[0]

        # Calculate recovery loss.
        loss = 0.
        dec_out = decoder(encode)  # (B, C, H, W)
        loss += self.lambda_ * F.mse_loss(dec_out, tgt)

        loss += (1 - self.lambda_) * (1 - torch.mean(ssim(dec_out, tgt, data_range=1, size_average=False)))

        return loss 


class RobustDAA(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'RobustDAA_BCELoss'
        self.loss_func = nn.BCEWithLogitsLoss()

    def EncoderDecoder(self, models, enc_metas):
        encoder, decoder, attention = models

        enc_output = encoder(*enc_metas)
        dec_output = decoder(enc_output)
        x_f = torch.mul(attention(*enc_metas), dec_output)
        output = dec_output + x_f

        return output
    
    def forward(self, models, enc_metas, rec_metas):

        output = self.EncoderDecoder(models, enc_metas)

        trg = rec_metas[1]
        loss = self.loss_func(output, trg) # trg中的值为非负数
        
        return loss

class TrajODE(nn.Module):
    def __init__(self, flat_valid, input_cols, decode_size, embed_dim, num_class, cnf_flag=True, elbo_weight=5e-4):
        super().__init__()
        self.name = f'TrajODE-ew{elbo_weight}'

        self.flat_valid = flat_valid
        self.input_cols = input_cols
        self.decode_size = decode_size
        self.elbo_weight = elbo_weight

        self.cnf_flag = cnf_flag

        self.start_token = nn.Parameter(torch.randn(decode_size).float(), requires_grad=True)

        # self.classify_linear = nn.Linear(embed_dim, num_class)

    def forward(self, models, enc_metas, rec_metas):
        encoder, decoder = models

        encode = encoder(*enc_metas)

        trips, lengths = rec_metas[:2]
        B, L, E = trips.shape
        tgt_latent = torch.cat([repeat(self.start_token, 'E -> B 1 E', B=B),
                                trips[:, :-1, self.input_cols]], 1)

        loss = 0.0
        # Calculate recovery loss.
        dec_out, kld_loss = decoder(tgt_latent, encode)  # (B, L, trip_fea_num)
        pred, label = DDPM._flat(self.flat_valid, dec_out, trips, length=lengths)
        loss += self.elbo_weight * F.mse_loss(pred, label)  # reconstruction likelihood

        # KL divergences between p(z) and q(z_0|x)
        # loss -= 0.5 * torch.sum(1 + decode_logvar - decode_mean.pow(2) - decode_logvar.exp())
        print("recon loss: ", loss)
        loss += kld_loss
        print("kld loss: ", kld_loss)

        # Normalizing flow Jacobi
        # pass

        return loss


class MAERR(nn.Module):
    def __init__(self, mask_ratio1=0.7, mask_ratio2=0.8):
        super(MAERR, self).__init__()
        self.mask_ratio1 = mask_ratio1
        self.mask_ratio2 = mask_ratio2
        self.name = 'MAERR'

    def forward(self, models, enc_metas, rec_metas):
        model, = models
        loss_cls, loss_rec, loss_rr, _, _, _, _ = model.forward_loss(*enc_metas)
        loss = loss_cls + loss_rec + loss_rr

        return loss
