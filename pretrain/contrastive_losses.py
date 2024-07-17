import torch
import torch.nn.functional as F
from torch import nn


class MEC(nn.Module):
    """
    Maximum Entropy Coding loss for contrastive between two views.
    Liu X, Wang Z, Li Y, et al. Self-Supervised Learning via Maximum Entropy Coding. NeuralIPS 2022.
    """

    def __init__(self, embed_dim, hidden_size, n, teachers):
        super().__init__()

        # The predictor is applied on top of the encoder as a non-linear casting.
        self.predictor = nn.Sequential(nn.Linear(embed_dim, hidden_size, bias=False),
                                       nn.BatchNorm1d(hidden_size),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(hidden_size, embed_dim))

        # The teachers can be symmetric or asymmetric models of the encoders.
        self.teachers = nn.ModuleList(teachers)
        self.n = n

        self.name = f'mec-e{embed_dim}-h{hidden_size}-n{n}'

    def forward(self, encoders, *metas, lamda_inv=1):
        z1, z2 = (encoder(*metas) for encoder in encoders)
        z1, z2 = self.predictor(z1), self.predictor(z2)
        with torch.no_grad():
            # The teachers are totally detached from the gradient updating.
            # Instead, they can be updated using the momentum trainer.
            p1, p2 = (teacher(*metas) for teacher in self.teachers)
        p1 = p1.detach()
        p2 = p2.detach()

        # Symmetric loss between two views.
        batch_size = metas[0][0].shape[0]
        loss = (self.mec(p1, z2, lamda_inv) + self.mec(p2, z1, lamda_inv)) * 0.5 / batch_size
        loss = -1 * loss * lamda_inv
        return loss

    def mec(self, view1, view2, lamda_inv):
        view1, view2 = F.normalize(view1), F.normalize(view2)
        c = torch.mm(view1, view2.transpose(0, 1)) / lamda_inv  # (B, B)
        power = c
        sum_p = torch.zeros_like(power)
        for k in range(1, self.n+1):
            if k > 1:
                power = torch.mm(power, c)
            if (k + 1) % 2 == 0:
                sum_p += power / k
            else:
                sum_p -= power / k
        trace = torch.trace(sum_p)
        return trace


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.
    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113
    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.
    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.
    Returns:
         Value of the InfoNCE Loss.
     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

        self.name = f'infonce-temp{temperature}'

    def forward(self, encoders, *metas):
        query, pos_key = (encoder(*meta) for encoder, meta in zip(encoders, metas))  # (B, E)
        return InfoNCE.info_nce(query, pos_key, pos_key,
                                temperature=self.temperature,
                                reduction=self.reduction,
                                negative_mode=self.negative_mode)

    @staticmethod
    def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        # Check input dimensionality.
        if query.dim() != 2:
            raise ValueError('<query> must have 2 dimensions.')
        if positive_key.dim() != 2:
            raise ValueError('<positive_key> must have 2 dimensions.')
        if negative_keys is not None:
            if negative_mode == 'unpaired' and negative_keys.dim() != 2:
                raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
            if negative_mode == 'paired' and negative_keys.dim() != 3:
                raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

        # Check matching number of samples.
        if len(query) != len(positive_key):
            raise ValueError('<query> and <positive_key> must must have the same number of samples.')
        if negative_keys is not None:
            if negative_mode == 'paired' and len(query) != len(negative_keys):
                raise ValueError(
                    "If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

        # Embedding vectors should have same number of components.
        if query.shape[-1] != positive_key.shape[-1]:
            raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
        if negative_keys is not None:
            if query.shape[-1] != negative_keys.shape[-1]:
                raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

        # Normalize to unit vectors
        query, positive_key, negative_keys = InfoNCE.normalize(query, positive_key, negative_keys)
        if negative_keys is not None:
            # Explicit negative keys

            # Cosine between positive pairs
            positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

            if negative_mode == 'unpaired':
                # Cosine between all query-negative combinations
                negative_logits = query @ InfoNCE.transpose(negative_keys)

            elif negative_mode == 'paired':
                query = query.unsqueeze(1)
                negative_logits = query @ InfoNCE.transpose(negative_keys)
                negative_logits = negative_logits.squeeze(1)

            # First index in last dimension are the positive samples
            logits = torch.cat([positive_logit, negative_logits], dim=1)
            labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
        else:
            # Negative keys are implicitly off-diagonal positive keys.

            # Cosine between all combinations
            logits = query @ InfoNCE.transpose(positive_key)

            # Positive keys are the entries on the diagonal
            labels = torch.arange(len(query), device=query.device)

        return F.cross_entropy(logits / temperature, labels, reduction=reduction)

    @staticmethod
    def transpose(x):
        return x.transpose(-2, -1)

    @staticmethod
    def normalize(*xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]


class SimCLR(nn.Module):
    """
    SimCLR liked contrastive loss
    Chen T, Kornblith S, Norouzi M, et al. A simple framework for contrastive learning of visual representations. ICML 2020
    """

    def __init__(self, embed_dim, similarity='inner', temperature=0.05):
        super().__init__()

        self.similarity = similarity
        self.temperature = temperature
        self.predictor = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                       nn.Tanh())
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

        self.name = f'simclr-e{embed_dim}-sim{similarity}-t{temperature}'

    def forward(self, encoders, *metas, common_meta=[]):
        if len(encoders) == 2:
            encoder1 = encoders[0]
            encoder2 = encoders[1]
        elif len(encoders) == 1:
            encoder1 = encoder2 = encoders[0]
        else:
            raise NotImplementedError("length of encoders must be 1 or 2.")

        in_metas = []
        for m in metas:
            in_metas += m
        side_length = len(in_metas) // 2
        z1 = encoder1(*in_metas[:side_length], *common_meta, pretrain=False)
        z2 = encoder2(*in_metas[side_length:], *common_meta, pretrain=False)

        z1, z2 = self.predictor(z1), self.predictor(z2)  # (B, embed_dim)

        batch_size, embed_dim = z1.shape
        features = torch.cat([z1, z2], dim=0)  # (batch_size * 2, embed_dim)
        features = F.normalize(features, dim=1)

        labels = torch.cat([torch.arange(batch_size, device=z1.device) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        if self.similarity == 'inner':
            similarity_matrix = torch.matmul(features, features.T)
        elif self.similarity == 'cosine':
            similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
        else:
            similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool, device=features.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # [batch_size * 2, 1]

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)  # [batch_size * 2, 2N-2]

        logits = torch.cat([positives, negatives], dim=1)  # (batch_size * 2, batch_size * 2 - 1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)  # (batch_size * 2, 1)
        logits = logits / self.temperature
        logits = torch.where(torch.isnan(logits), torch.full_like(logits, 1e-4), logits)
        labels[0] = 1

        loss_res = self.criterion(logits, labels)

        return loss_res


class GeoConstrainWord2Vec(nn.Module):
    def __init__(self, **kwargs):
        super(GeoConstrainWord2Vec, self).__init__()
        self.name = 'GeoConstrainWord2Vec'

    def forward(self, models, metas):
        assert len(models) == 1
        emb_sd, emb_m, emb_neg = models[0].forward_train(*metas)

        score = torch.sum(torch.mul(emb_sd, emb_m), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg, emb_sd.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)
