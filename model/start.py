import math
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from model.base import Encoder, Decoder
from model.transformer import PositionalEncoding


class GATLayer(torch.nn.Module):
    """
    A GAT layer, similar to https://arxiv.org/abs/1710.10903
    An implementation from START
    Base class for all implementations as there is much code that would otherwise be copy/pasted.
    """

    head_dim = 1

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, load_trans_prob=True):

        super().__init__()

        # Saving these as we'll need them in forward propagation in children layers (imp1/2/3)
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection
        self.load_trans_prob = load_trans_prob

        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
        #
        # You can treat this one matrix as num_of_heads independent W matrices
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        if self.load_trans_prob:
            self.linear_proj_tran_prob = nn.Linear(1, num_of_heads * num_out_features, bias=False)

        # After we concatenate target node (node i) and source node (node j) we apply the additive scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.

        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        if self.load_trans_prob:
            self.scoring_trans_prob = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar, on this one)
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        #
        # End of trainable weights
        #
        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the log-softmax along the last dimension
        self.activation = activation
        # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
        # and for attention coefficients. Functionality-wise it's the same as using independent modules.
        self.dropout = nn.Dropout(p=dropout_prob)

        self.init_params()

    def init_params(self):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.

        """
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        if self.load_trans_prob:
            nn.init.xavier_uniform_(self.linear_proj_tran_prob.weight)
            nn.init.xavier_uniform_(self.scoring_trans_prob)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, in_nodes_features, out_nodes_features):

        # if the tensor is not contiguously stored in memory we'll get an error after we try to do certain ops like view
        # only imp1 will enter this one
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


class GATLayerImp3(GATLayer):
    """
    An implementation from libcity

    Implementation #3 was inspired by PyTorch Geometric: https://github.com/rusty1s/pytorch_geometric

    But, it's hopefully much more readable! (and of similar performance)

    It's suitable for both transductive and inductive settings. In the inductive setting we just merge the graphs
    into a single graph with multiple components and this layer is agnostic to that fact! <3

    """

    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    nodes_dim = 0  # node dimension/axis
    head_dim = 1  # attention head dimension/axis

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, load_trans_prob=True):

        # Delegate initialization to the base class
        super().__init__(num_in_features, num_out_features, num_of_heads, concat, activation, dropout_prob,
                         add_skip_connection, bias, load_trans_prob)

    def forward(self, data):
        #
        # Step 1: Linear Projection + regularization
        #
        in_nodes_features, edge_index, edge_prob = data  # unpack data edge_prob=(E, 1)
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        # Note: for Cora features are already super sparse so it's questionable how much this actually helps
        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        nodes_features_proj = self.dropout(
            nodes_features_proj)  # in the official GAT imp they did dropout here as well

        if self.load_trans_prob:
            # shape = (E, 1) * (1, NH*FOUT) -> (E, NH, FOUT) where NH - number of heads, FOUT - num of output features
            trans_prob_proj = self.linear_proj_tran_prob(edge_prob).view(
                -1, self.num_of_heads, self.num_out_features)  # (E, NH, FOUT)
            trans_prob_proj = self.dropout(trans_prob_proj)
        #
        # Step 2: Edge attention calculation
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, FOUT) -> (N, NH) because sum squeezes the last dimension
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(
            scores_source, scores_target, nodes_features_proj, edge_index)
        if self.load_trans_prob:
            # shape = (E, NH, FOUT) * (1, NH, FOUT) -> (E, NH, FOUT) -> (E, NH)
            scores_trans_prob = (trans_prob_proj * self.scoring_trans_prob).sum(dim=-1)  # (E, NH)
            scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted + scores_trans_prob)
        else:
            scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        # shape = (E, NH, 1)
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim],
                                                              num_of_nodes)
        # Add stochasticity to neighborhood aggregation
        attentions_per_edge = self.dropout(attentions_per_edge)

        #
        # Step 3: Neighborhood aggregation
        #

        # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
        # shape = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT), 1 gets broadcast into FOUT
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge

        # This part sums up weighted and projected neighborhood feature vectors for every target node
        # shape = (N, NH, FOUT)
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index,
                                                      in_nodes_features, num_of_nodes)

        #
        # Step 4: Residual/skip connections, concat and bias
        #
        out_nodes_features = self.skip_concat_bias(in_nodes_features, out_nodes_features)
        return (out_nodes_features, edge_index, edge_prob)

    #
    # Helper functions (without comments there is very little code so don't be scared!)
    #

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        """
        As the fn name suggest it does softmax over the neighborhoods. Example: say we have 5 nodes in a graph.
        Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
        into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
        in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
        (where 1-3 is overloaded notation it represents the edge 1-3 and it's (exp) score) and similarly for 2-3 and 3-3
         i.e. for this neighborhood we don't care about other edge scores that include nodes 4 and 5.

        Note:
        Subtracting the max value from logits doesn't change the end result but it improves the numerical stability
        and it's a fairly common "trick" used in pretty much every deep learning libcity.
        Check out this link for more details:

        https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning

        """
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        scores_per_edge = scores_per_edge - scores_per_edge.max()  # (E, NH)
        exp_scores_per_edge = scores_per_edge.exp()  # softmax, (E, NH)

        # Calculate the denominator. shape = (E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index,
                                                                                num_of_nodes)

        # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
        # possibility of the computer rounding a very small number all the way to 0.
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
        # target index)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

        # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
        # all the locations where the source nodes pointed to i (as dictated by the target index)
        # shape = (N, NH) -> (E, NH)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(
            nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim],
                                                        nodes_features_proj_lifted_weighted)
        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).

        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        # E -> (E, NH)
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)


class GAT(nn.Module):
    """An implementation from libcity"""

    def __init__(self, d_model, in_feature, num_heads_per_layer, num_features_per_layer,
                 add_skip_connection=True, bias=True, dropout=0.6, load_trans_prob=True, avg_last=True):
        super().__init__()
        self.d_model = d_model
        assert len(num_heads_per_layer) == len(num_features_per_layer), f'Enter valid arch params.'

        num_features_per_layer = [in_feature] + num_features_per_layer
        num_heads_per_layer = [1] + num_heads_per_layer  # trick - so that I can nicely create GAT layers below
        if avg_last:
            assert num_features_per_layer[-1] == d_model
        else:
            assert num_features_per_layer[-1] * num_heads_per_layer[-1] == d_model
        num_of_layers = len(num_heads_per_layer) - 1

        gat_layers = []  # collect GAT layers
        for i in range(num_of_layers):
            if i == num_of_layers - 1:
                if avg_last:
                    concat_input = False
                else:
                    concat_input = True
            else:
                concat_input = True
            layer = GATLayerImp3(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],  # consequence of concatenation
                num_out_features=num_features_per_layer[i+1],
                num_of_heads=num_heads_per_layer[i+1],
                concat=concat_input,  # last GAT layer does mean avg, the others do concat
                activation=nn.ELU() if i < num_of_layers - 1 else None,  # last layer just outputs raw scores
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                load_trans_prob=load_trans_prob
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(
            *gat_layers,
        )

    def forward(self, node_features, edge_index_input, edge_prob_input, x):
        """

        Args:
            node_features: (vocab_size, fea_dim)
            edge_index_input: (2, E)
            edge_prob_input: (E, 1)
            x: (B, T)

        Returns:
            (B, T, d_model)

        """
        data = (node_features, edge_index_input, edge_prob_input)
        (node_fea_emb, edge_index, edge_prob) = self.gat_net(data)  # (vocab_size, num_channels[-1]), (2, E)
        batch_size, seq_len = x.shape
        node_fea_emb = node_fea_emb.expand((batch_size, -1, -1))  # (B, vocab_size, d_model)
        node_fea_emb = node_fea_emb.reshape(-1, self.d_model)  # (B * vocab_size, d_model)
        x = x.reshape(-1, 1).squeeze(1)  # (B * T,)
        out_node_fea_emb = node_fea_emb[x].reshape(batch_size, seq_len, self.d_model)  # (B, T, d_model)
        return out_node_fea_emb  # (B, T, d_model)


def drop_path_func(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_func(x, self.drop_prob, self.training)


class ProEmbedding(nn.Module):

    def __init__(self, d_model, num_embeds, dis_feats, road_feat, num_tokens=None, token_feat=None, dropout=None,
                 add_gat=False, gat_num_features=None, gat_num_heads=None, gat_dropout=None):
        super().__init__()

        self.d_model = d_model
        self.num_embeds = num_embeds
        self.dis_feats = dis_feats
        self.road_feat = road_feat
        self.num_tokens = num_tokens
        self.token_feat = token_feat
        self.dropout = dropout
        self.add_gat = add_gat

        self.position_embedding = PositionalEncoding(embed_size=d_model, max_len=512)

        if add_gat:
            self.gat_encoder = GAT(d_model=d_model, in_feature=d_model,
                                       num_heads_per_layer=gat_num_heads,
                                       num_features_per_layer=gat_num_features,
                                       add_skip_connection=True, bias=True, dropout=gat_dropout,
                                       load_trans_prob=True, avg_last=True)

        if dis_feats is not None and len(dis_feats) > 0:
            assert len(dis_feats) == len(num_embeds), \
                'length of num_embeds list should be equal to the number of discrete features.'
            self.dis_embeds = nn.ModuleList([nn.Embedding(num_embed, d_model) for num_embed in num_embeds])

        if num_tokens and token_feat is not None:
            self.token_embedding = nn.Embedding(num_tokens + 2, d_model)

        if dropout:
            self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, trip, edge_index_input, edge_prob_input, pretrain=False):
        """
        Args:
            trip: (B, max_len, fea_dim) ['tod', 'road', 'road_prop', 'lng', 'lat', 'weekday', 'seq_i', 'seconds']
            edge_index_input: (2, num_edges)
            edge_prob_input: (num_edges, 1)
            pretrain: True or False
        """

        # seconds -> minutes
        trip[:, :, 7] = torch.round(trip[:, :, 7] / 60) % 1440

        dis_x = torch.stack([embed(trip[..., feat].long())
                             for feat, embed in zip(self.dis_feats, self.dis_embeds)], -1).sum(-1)
        out = dis_x

        if self.num_tokens and self.token_feat is not None:
            if pretrain:
                token_x = self.token_embedding(trip[..., self.token_feat].long())
                if self.add_gat:  # simplified (without segment properties)
                    token_x += self.gat_encoder(self.token_embedding.weight.detach(), edge_index_input, edge_prob_input,
                                            trip[..., self.token_feat].long())
            else:
                token_x = self.token_embedding(trip[..., self.road_feat].long())
                if self.add_gat:  # simplified (without segment properties)
                    token_x += self.gat_encoder(self.token_embedding.weight.detach(), edge_index_input, edge_prob_input,
                                            trip[..., self.road_feat].long())
            out += token_x

        if self.dropout:
            out = self.dropout_layer(out)
        out += self.position_embedding(out)

        return out


class BERTEmbedding(nn.Module):

    def __init__(self, d_model, num_embed, dropout=0.1, add_time_in_day=False, add_day_in_week=False,
                 add_pe=True, node_fea_dim=10, add_gat=True,
                 gat_heads_per_layer=None, gat_features_per_layer=None, gat_dropout=0.6,
                 load_trans_prob=True, avg_last=True):
        """

        Args:
            vocab_size: total vocab size
            d_model: embedding size of token embedding
            dropout: dropout rate
        """
        super().__init__()
        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week
        self.add_pe = add_pe
        self.add_gat = add_gat

        if self.add_gat:
            self.token_embedding = GAT(d_model=d_model, in_feature=node_fea_dim,
                                   num_heads_per_layer=gat_heads_per_layer,
                                   num_features_per_layer=gat_features_per_layer,
                                   add_skip_connection=True, bias=True, dropout=gat_dropout,
                                   load_trans_prob=load_trans_prob, avg_last=avg_last)
        else:
            self.token_embedding = nn.Embedding(num_embed + 2, d_model)
        if self.add_pe:
            self.position_embedding = PositionalEncoding(embed_size=d_model, max_len=512)
        if self.add_time_in_day:
            self.daytime_embedding = nn.Embedding(1441, d_model, padding_idx=0)
        if self.add_day_in_week:
            self.weekday_embedding = nn.Embedding(8, d_model, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, sequence, position_ids=None, graph_dict=None):
        """

        Args:
            sequence: (B, T, F) [tod, road, road_prop, lng, lat, weekday, seq_i, offset, minute]
            position_ids: (B, T) or None
            graph_dict(dict): including:
                in_lap_mx: (vocab_size, lap_dim)
                out_lap_mx: (vocab_size, lap_dim)
                indegree: (vocab_size, )
                outdegree: (vocab_size, )

        Returns:
            (B, T, d_model)

        """
        if self.add_gat:
            x = self.token_embedding(node_features=graph_dict['node_features'],
                                     edge_index_input=graph_dict['edge_index'],
                                     edge_prob_input=graph_dict['loc_trans_prob'],
                                     x=sequence[:, :, 1])  # (B, T, d_model)
        else:
            x = self.token_embedding(sequence[:, :, 1].long())  # (B, T, d_model)
        if self.add_pe:
            x += self.position_embedding(x, position_ids)  # (B, T, d_model)
        if self.add_time_in_day:
            x += self.daytime_embedding(sequence[:, :, 7].long())  # (B, T, d_model)
        if self.add_day_in_week:
            x += self.weekday_embedding(sequence[:, :, 5].long())  # (B, T, d_model)
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):

    def __init__(self, num_heads, d_model, dim_out, attn_drop=0., proj_drop=0.,
                 add_cls=True, device=torch.device('cpu'), add_temporal_bias=True,
                 temporal_bias_dim=64, use_mins_interval=False):
        super().__init__()
        assert d_model % num_heads == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.device = device
        self.add_cls = add_cls
        self.scale = self.d_k ** -0.5  # 1/sqrt(dk)
        self.add_temporal_bias = add_temporal_bias
        self.temporal_bias_dim = temporal_bias_dim
        self.use_mins_interval = use_mins_interval

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.dropout = nn.Dropout(p=attn_drop)

        self.proj = nn.Linear(d_model, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.add_temporal_bias:
            if self.temporal_bias_dim != 0 and self.temporal_bias_dim != -1:
                self.temporal_mat_bias_1 = nn.Linear(1, self.temporal_bias_dim, bias=True)
                self.temporal_mat_bias_2 = nn.Linear(self.temporal_bias_dim, 1, bias=True)
            elif self.temporal_bias_dim == -1:
                self.temporal_mat_bias = nn.Parameter(torch.Tensor(1, 1))
                nn.init.xavier_uniform_(self.temporal_mat_bias)

    def forward(self, x, padding_masks, future_mask=True, output_attentions=False, batch_temporal_mat=None):
        """

        Args:
            x: (B, T, d_model)
            padding_masks: (B, 1, T, T) padding_mask
            future_mask: True/False
            batch_temporal_mat: (B, T, T)

        Returns:

        """
        batch_size, seq_len, d_model = x.shape

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # l(x) --> (B, T, d_model)
        # l(x).view() --> (B, T, head, d_k)
        query, key, value = [l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (x, x, x))]
        # q, k, v --> (B, head, T, d_k)

        # 2) Apply attention on all the projected vectors in batch.
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale  # (B, head, T, T)

        if self.add_temporal_bias:
            if self.use_mins_interval:
                batch_temporal_mat = 1.0 / torch.log(
                    torch.exp(torch.tensor(1.0).to(self.device)) +
                    (batch_temporal_mat / torch.tensor(60.0).to(self.device)))
            else:
                batch_temporal_mat = 1.0 / torch.log(
                    torch.exp(torch.tensor(1.0).to(self.device)) + batch_temporal_mat)
            if self.temporal_bias_dim != 0 and self.temporal_bias_dim != -1:
                batch_temporal_mat = self.temporal_mat_bias_2(F.leaky_relu(
                    self.temporal_mat_bias_1(batch_temporal_mat.unsqueeze(-1)),
                    negative_slope=0.2)).squeeze(-1)  # (B, T, T)
            if self.temporal_bias_dim == -1:
                batch_temporal_mat = batch_temporal_mat * self.temporal_mat_bias.expand((1, seq_len, seq_len))
            batch_temporal_mat = batch_temporal_mat.unsqueeze(1)  # (B, 1, T, T)
            scores += batch_temporal_mat  # (B, 1, T, T)

        if padding_masks is not None:
            scores.masked_fill_(padding_masks == 0, float('-inf'))

        if future_mask:
            mask_postion = torch.triu(torch.ones((1, seq_len, seq_len)), diagonal=1).bool().to(self.device)
            if self.add_cls:
                mask_postion[:, 0, :] = 0
            scores.masked_fill_(mask_postion, float('-inf'))

        p_attn = F.softmax(scores, dim=-1)  # (B, head, T, T)
        p_attn = self.dropout(p_attn)
        out = torch.matmul(p_attn, value)  # (B, head, T, d_k)

        # 3) "Concat" using a view and apply a final linear.
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)  # (B, T, d_model)
        out = self.proj(out)  # (B, T, N, D)
        out = self.proj_drop(out)
        if output_attentions:
            return out, p_attn  # (B, T, dim_out), (B, head, T, T)
        else:
            return out, None  # (B, T, dim_out)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        """Position-wise Feed-Forward Networks
        Args:
            in_features:
            hidden_features:
            out_features:
            act_layer:
            drop:
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, d_model, attn_heads, feed_forward_hidden, drop_path,
                 attn_drop, dropout, type_ln='pre', add_cls=True,
                 device=torch.device('cpu'), add_temporal_bias=True,
                 temporal_bias_dim=64, use_mins_interval=False):
        """

        Args:
            d_model: hidden size of transformer
            attn_heads: head sizes of multi-head attention
            feed_forward_hidden: feed_forward_hidden, usually 4*d_model
            drop_path: encoder dropout rate
            attn_drop: attn dropout rate
            dropout: dropout rate
            type_ln:
        """

        super().__init__()
        self.attention = MultiHeadedAttention(num_heads=attn_heads, d_model=d_model, dim_out=d_model,
                                              attn_drop=attn_drop, proj_drop=dropout, add_cls=add_cls,
                                              device=device, add_temporal_bias=add_temporal_bias,
                                              temporal_bias_dim=temporal_bias_dim,
                                              use_mins_interval=use_mins_interval)
        self.mlp = Mlp(in_features=d_model, hidden_features=feed_forward_hidden,
                       out_features=d_model, act_layer=nn.GELU, drop=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.type_ln = type_ln

    def forward(self, x, padding_masks, future_mask=True, output_attentions=False, batch_temporal_mat=None):
        """

        Args:
            x: (B, T, d_model)
            padding_masks: (B, 1, T, T)
            future_mask: True/False
            batch_temporal_mat: (B, T, T)

        Returns:
            (B, T, d_model)

        """
        if self.type_ln == 'pre':
            attn_out, attn_score = self.attention(self.norm1(x), padding_masks=padding_masks,
                                                  future_mask=future_mask, output_attentions=output_attentions,
                                                  batch_temporal_mat=batch_temporal_mat)
            x = x + self.drop_path(attn_out)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif self.type_ln == 'post':
            attn_out, attn_score = self.attention(x, padding_masks=padding_masks,
                                                  future_mask=future_mask, output_attentions=output_attentions,
                                                  batch_temporal_mat=batch_temporal_mat)
            x = self.norm1(x + self.drop_path(attn_out))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            raise ValueError('Error type_ln {}'.format(self.type_ln))
        return x, attn_score


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, config, data_feature):
        """
        Args:
        """
        super().__init__()

        self.config = config

        self.vocab_size = data_feature.get('vocab_size')
        self.node_fea_dim = data_feature.get('node_fea_dim')

        self.num_tokens = self.config.get('num_roads', 2505)
        self.num_embeds = self.config.get('num_embeds', [])
        self.dis_feats = self.config.get('dis_feats', [])
        self.token_feat = self.config.get('token_feat', 8)
        self.road_feat = self.config.get('road_feat', 1)

        self.d_model = self.config.get('d_model', 768)
        self.n_layers = self.config.get('n_layers', 12)
        self.attn_heads = self.config.get('attn_heads', 12)
        self.mlp_ratio = self.config.get('mlp_ratio', 4)
        self.dropout = self.config.get('dropout', 0.)
        self.drop_path = self.config.get('drop_path', 0.)
        self.lape_dim = self.config.get('lape_dim', 256)
        self.attn_drop = self.config.get('attn_drop', 0.1)
        self.type_ln = self.config.get('type_ln', 'pre')
        self.future_mask = self.config.get('future_mask', False)
        self.add_cls = self.config.get('add_cls', False)
        self.device = self.config.get('device', torch.device('cpu'))
        self.cutoff_row_rate = self.config.get('cutoff_row_rate', 0.2)
        self.cutoff_column_rate = self.config.get('cutoff_column_rate', 0.2)
        self.cutoff_random_rate = self.config.get('cutoff_random_rate', 0.2)
        self.sample_rate = self.config.get('sample_rate', 0.2)
        self.device = self.config.get('device', torch.device('cpu'))
        self.add_time_in_day = self.config.get('add_time_in_day', True)
        self.add_day_in_week = self.config.get('add_day_in_week', True)
        self.add_pe = self.config.get('add_pe', True)
        self.add_gat = self.config.get('add_gat', False)
        self.gat_heads_per_layer = self.config.get('gat_heads_per_layer', [8, 1])
        self.gat_features_per_layer = self.config.get('gat_features_per_layer', [16, self.d_model])
        self.gat_dropout = self.config.get('gat_dropout', 0.6)
        self.gat_avg_last = self.config.get('gat_avg_last', True)
        self.load_trans_prob = self.config.get('load_trans_prob', False)
        self.add_temporal_bias = self.config.get('add_temporal_bias', True)
        self.temporal_bias_dim = self.config.get('temporal_bias_dim', 64)
        self.use_mins_interval = self.config.get('use_mins_interval', False)

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = self.d_model * self.mlp_ratio

        # embedding for BERT, sum of ... embeddings
        # self.embedding = BERTEmbedding(d_model=self.d_model, num_embed=self.num_tokens, dropout=self.dropout,
        #                                add_time_in_day=self.add_time_in_day, add_day_in_week=self.add_day_in_week,
        #                                add_pe=self.add_pe, node_fea_dim=self.node_fea_dim, add_gat=self.add_gat,
        #                                gat_heads_per_layer=self.gat_heads_per_layer,
        #                                gat_features_per_layer=self.gat_features_per_layer, gat_dropout=self.gat_dropout,
        #                                load_trans_prob=self.load_trans_prob, avg_last=self.gat_avg_last)
        self.embedding = ProEmbedding(d_model=self.d_model, num_embeds=self.num_embeds, dis_feats=self.dis_feats,
                                      road_feat=self.road_feat, num_tokens=self.vocab_size, token_feat=self.token_feat,
                                      dropout=self.dropout, add_gat=self.add_gat, gat_num_features=self.gat_features_per_layer,
                                      gat_num_heads=self.gat_heads_per_layer, gat_dropout=self.gat_dropout)

        # multi-layers transformer blocks, deep network
        enc_dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.n_layers)]  # stochastic depth decay rule
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model=self.d_model, attn_heads=self.attn_heads,
                              feed_forward_hidden=self.feed_forward_hidden, drop_path=enc_dpr[i],
                              attn_drop=self.attn_drop, dropout=self.dropout,
                              type_ln=self.type_ln, add_cls=self.add_cls,
                              device=self.device, add_temporal_bias=self.add_temporal_bias,
                              temporal_bias_dim=self.temporal_bias_dim,
                              use_mins_interval=self.use_mins_interval) for i in range(self.n_layers)])

    def _shuffle_position_ids(self, x, padding_masks, position_ids):
        batch_size, seq_len, feat_dim = x.shape
        if position_ids is None:
            position_ids = torch.arange(512).expand((batch_size, -1))[:, :seq_len].to(device=self.device)

        # shuffle position_ids
        shuffled_pid = []
        for bsz_id in range(batch_size):
            sample_pid = position_ids[bsz_id]  # (512, )
            sample_mask = padding_masks[bsz_id]  # (seq_length, )
            num_tokens = sample_mask.sum().int().item()
            indexes = list(range(num_tokens))
            random.shuffle(indexes)
            rest_indexes = list(range(num_tokens, seq_len))
            total_indexes = indexes + rest_indexes
            shuffled_pid.append(torch.index_select(sample_pid, 0, torch.tensor(total_indexes).to(
                device=self.device)).unsqueeze(0))
        shuffled_pid = torch.cat(shuffled_pid, 0).to(device=self.device)   # (batch_size, seq_length)
        return shuffled_pid

    def _sample_span(self, x, padding_masks, sample_rate=0.2):
        batch_size, seq_len, feat_dim = x.shape

        if sample_rate > 0:
            true_seq_len = padding_masks.sum(1).cpu().numpy()
            mask = []
            for true_len in true_seq_len:
                sample_len = max(int(true_len * (1 - sample_rate)), 1)
                start_id = np.random.randint(0, high=true_len - sample_len + 1)
                tmp = [1] * seq_len
                for idx in range(start_id, start_id + sample_len):
                    tmp[idx] = 0
                mask.append(tmp)
            mask = torch.ByteTensor(mask).bool().to(self.device)
            x = x.masked_fill(mask.unsqueeze(-1), value=0.)
            padding_masks = padding_masks.masked_fill(mask, value=0)

        return x, padding_masks

    def _cutoff_embeddings(self, embedding_output, padding_masks, direction, cutoff_rate=0.2):
        batch_size, seq_len, d_model = embedding_output.shape
        cutoff_embeddings = []
        for bsz_id in range(batch_size):
            sample_embedding = embedding_output[bsz_id]
            sample_mask = padding_masks[bsz_id]
            if direction == "row":
                num_dimensions = sample_mask.sum().int().item()  # number of tokens
                dim_index = 0
            elif direction == "column":
                num_dimensions = d_model
                dim_index = 1
            elif direction == "random":
                num_dimensions = sample_mask.sum().int().item() * d_model
                dim_index = 0
            else:
                raise ValueError(f"direction should be either row or column, but got {direction}")
            num_cutoff_indexes = int(num_dimensions * cutoff_rate)
            if num_cutoff_indexes < 0 or num_cutoff_indexes > num_dimensions:
                raise ValueError(f"number of cutoff dimensions should be in (0, {num_dimensions}), but got {num_cutoff_indexes}")
            indexes = list(range(num_dimensions))
            random.shuffle(indexes)
            cutoff_indexes = indexes[:num_cutoff_indexes]
            if direction == "random":
                sample_embedding = sample_embedding.reshape(-1)  # (seq_length * d_model, )
            cutoff_embedding = torch.index_fill(sample_embedding, dim_index, torch.tensor(
                cutoff_indexes, dtype=torch.long).to(device=self.device), 0.0)
            if direction == "random":
                cutoff_embedding = cutoff_embedding.reshape(seq_len, d_model)
            cutoff_embeddings.append(cutoff_embedding.unsqueeze(0))
        cutoff_embeddings = torch.cat(cutoff_embeddings, 0).to(device=self.device)  # (batch_size, seq_length, d_model)
        assert cutoff_embeddings.shape == embedding_output.shape
        return cutoff_embeddings

    def _shuffle_embeddings(self, embedding_output, padding_masks):
        batch_size, seq_len, d_model = embedding_output.shape
        shuffled_embeddings = []
        for bsz_id in range(batch_size):
            sample_embedding = embedding_output[bsz_id]  # (seq_length, d_model)
            sample_mask = padding_masks[bsz_id]  # (seq_length, )
            num_tokens = sample_mask.sum().int().item()
            indexes = list(range(num_tokens))
            random.shuffle(indexes)
            rest_indexes = list(range(num_tokens, seq_len))
            total_indexes = indexes + rest_indexes
            shuffled_embeddings.append(torch.index_select(sample_embedding, 0, torch.tensor(total_indexes).to(
                device=self.device)).unsqueeze(0))
        shuffled_embeddings = torch.cat(shuffled_embeddings, 0).to(device=self.device)  # (batch_size, seq_length, d_model)
        return shuffled_embeddings

    def forward(self, x, padding_masks, batch_temporal_mat=None, argument_methods=None,
                edge_index_input=None, edge_prob_input=None,
                output_hidden_states=False, output_attentions=False, pretrain=False):
        """
        Args:
            x: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """
        position_ids = None

        if argument_methods is not None:
            if 'shuffle_position' in argument_methods:
                position_ids = self._shuffle_position_ids(
                    x=x, padding_masks=padding_masks, position_ids=None)
            if 'span' in argument_methods:
                x, attention_mask = self._sample_span(
                    x=x, padding_masks=padding_masks, sample_rate=self.sample_rate)

        # embedding the indexed sequence to sequence of vectors
        # embedding_output = self.embedding(sequence=x, position_ids=position_ids,
        #                                   graph_dict=graph_dict, pretrain=pretrain)  # (B, T, d_model)
        embedding_output = self.embedding(trip=x, edge_index_input=edge_index_input, edge_prob_input=edge_prob_input,
                                          pretrain=pretrain)

        if argument_methods is not None:
            if 'cutoff_row' in argument_methods:
                embedding_output = self._cutoff_embeddings(
                    embedding_output=embedding_output, padding_masks=padding_masks,
                    direction='row', cutoff_rate=self.cutoff_row_rate)
            if 'cutoff_column' in argument_methods:
                embedding_output = self._cutoff_embeddings(
                    embedding_output=embedding_output, padding_masks=padding_masks,
                    direction='column', cutoff_rate=self.cutoff_column_rate)
            if 'cutoff_random' in argument_methods:
                embedding_output = self._cutoff_embeddings(
                    embedding_output=embedding_output, padding_masks=padding_masks,
                    direction='random', cutoff_rate=self.cutoff_random_rate)
            if 'shuffle_embedding' in argument_methods:
                embedding_output = self._shuffle_embeddings(
                    embedding_output=embedding_output, padding_masks=padding_masks)

        padding_masks_input = padding_masks.unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)  # (B, 1, T, T)
        # running over multiple transformer blocks
        all_hidden_states = [embedding_output] if output_hidden_states else None
        all_self_attentions = [] if output_attentions else None
        for transformer in self.transformer_blocks:
            embedding_output, attn_score = transformer.forward(
                x=embedding_output, padding_masks=padding_masks_input,
                future_mask=self.future_mask, output_attentions=output_attentions,
                batch_temporal_mat=batch_temporal_mat)  # (B, T, d_model)
            if output_hidden_states:
                all_hidden_states.append(embedding_output)
            if output_attentions:
                all_self_attentions.append(attn_score)
        return embedding_output, all_hidden_states, all_self_attentions  # (B, T, d_model), list of (B, T, d_model), list of (B, head, T, T)


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, d_model):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()

    def forward(self, features):
        x = self.dense(features)
        x = self.activation(x)
        return x


class BERTPooler(nn.Module):

    def __init__(self, config, data_feature):
        super().__init__()

        self.config = config

        self.pooling = self.config.get('pooling', 'mean')
        self.add_cls = self.config.get('add_cls', True)
        self.d_model = self.config.get('d_model', 768)
        self.linear = MLPLayer(d_model=self.d_model)

    def forward(self, bert_output, padding_masks, hidden_states=None):
        """
        Args:
            bert_output: (batch_size, seq_length, d_model) torch tensor of bert output
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
            hidden_states: list of hidden, (batch_size, seq_length, d_model)
        Returns:
            output: (batch_size, feat_dim)
        """
        token_emb = bert_output  # (batch_size, seq_length, d_model)
        if self.pooling == 'cls':
            if self.add_cls:
                return self.linear(token_emb[:, 0, :])  # (batch_size, feat_dim)
            else:
                raise ValueError('No use cls!')
        elif self.pooling == 'cls_before_pooler':
            if self.add_cls:
                return token_emb[:, 0, :]  # (batch_size, feat_dim)
            else:
                raise ValueError('No use cls!')
        elif self.pooling == 'mean':
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(
                token_emb.size()).float()  # (batch_size, seq_length, d_model)
            sum_embeddings = torch.sum(token_emb * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask  # (batch_size, feat_dim)
        elif self.pooling == 'max':
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(
                token_emb.size()).float()  # (batch_size, seq_length, d_model)
            token_emb[input_mask_expanded == 0] = float('-inf')  # Set padding tokens to large negative value
            max_over_time = torch.max(token_emb, 1)[0]
            return max_over_time  # (batch_size, feat_dim)
        elif self.pooling == "avg_first_last":
            first_hidden = hidden_states[0]  # (batch_size, seq_length, d_model)
            last_hidden = hidden_states[-1]  # (batch_size, seq_length, d_model)
            avg_emb = (first_hidden + last_hidden) / 2.0
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(
                avg_emb.size()).float()  # (batch_size, seq_length, d_model)
            sum_embeddings = torch.sum(avg_emb * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask  # (batch_size, feat_dim)
        elif self.pooling == "avg_top2":
            second_last_hidden = hidden_states[-2]  # (batch_size, seq_length, d_model)
            last_hidden = hidden_states[-1]  # (batch_size, seq_length, d_model)
            avg_emb = (second_last_hidden + last_hidden) / 2.0
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(
                avg_emb.size()).float()  # (batch_size, seq_length, d_model)
            sum_embeddings = torch.sum(avg_emb * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask  # (batch_size, feat_dim)
        else:
            raise ValueError('Error pooling type {}'.format(self.pooling))


class BERTEncoder(Encoder):
    """
    BERT Contrastive LM Encoder
    """

    def __init__(self, d_model, dis_feats, num_embeds, con_feats, hidden_size, num_heads, output_size,
                 vocab_size, road_feat, token_feat, num_layers, sampler, num_node_fea=7,
                 add_gat=False, gat_num_features=None, gat_num_heads=None, gat_dropout=None):
        super().__init__(sampler, 'BERT-' + ''.join(map(str, dis_feats + con_feats)) +
                         f'-d{d_model}-h{hidden_size}-l{num_layers}-h{num_heads}-o{output_size}')

        self.dis_feats = dis_feats
        self.con_feats = con_feats
        self.num_embeds = num_embeds
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.output_size = output_size

        bert_config = {
            "d_model": d_model,
            "n_layers": num_layers,
            "attn_heads": num_heads,
            "hidden_size": hidden_size,
            "pooling": "mean",
            "dis_feats": dis_feats,
            "num_embeds": num_embeds,
            "token_feat": token_feat,
            "road_feat": road_feat,
            "add_gat": add_gat,
            "gat_num_features": gat_num_features,
            "gat_num_heads": gat_num_heads,
            "gat_dropout": gat_dropout
        }
        bert_data_feature = {
            "vocab_size": vocab_size,
            "node_fea_dim": num_node_fea
        }
        self.encoder = BERT(bert_config, bert_data_feature)

        self.pooler = BERTPooler(bert_config, bert_data_feature)

    def forward(self, trip, valid_len, batch_temporal_mat=None, edge_index_input=None, edge_prob_input=None, pretrain=False):
        B, L, E_in = trip.shape
        src_key_padding_mask = repeat(torch.arange(end=L, device=trip.device),
                                      'L -> B L', B=B) >= repeat(valid_len, 'B -> B L', L=L)  # (B, L)

        encodes, _, _ = self.encoder(x=trip, padding_masks=src_key_padding_mask, batch_temporal_mat=batch_temporal_mat,
                                     edge_index_input=edge_index_input, edge_prob_input=edge_prob_input,
                                     argument_methods=None, output_hidden_states=False,
                                     output_attentions=False, pretrain=pretrain)

        if pretrain:
            return encodes

        encodes = self.pooler(bert_output=encodes, padding_masks=src_key_padding_mask,
                              hidden_states=None)

        return encodes
