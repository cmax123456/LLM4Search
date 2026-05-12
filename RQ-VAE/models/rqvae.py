import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import random
import collections
from .layers import MLPLayers
from .rq import ResidualVectorQuantizer
from .rq_kmeans import ResidualKmeansQuantizer


def _compute_auxiliary_loss(
    outer_con_losses,
    inner_triplet_losses,
    qd_align_loss,
    trade_off_inner_outer,
    inner_outer_layer_weight,
    use_outer_con_loss,
    use_inner_triplet_loss,
    use_qd_align_loss,
    qd_align_weight,
):
    aux_loss = 0.0

    if use_outer_con_loss or use_inner_triplet_loss:
        if outer_con_losses is not None and inner_triplet_losses is not None:
            for i in range(min(len(inner_outer_layer_weight), len(outer_con_losses), len(inner_triplet_losses))):
                w = inner_outer_layer_weight[i]
                o = trade_off_inner_outer[i] if i < len(trade_off_inner_outer) else 1.0
                term = 0.0
                if use_outer_con_loss:
                    term = term + o * outer_con_losses[i]
                if use_inner_triplet_loss:
                    term = term + (1 - o) * inner_triplet_losses[i]
                aux_loss = aux_loss + w * term
        elif inner_triplet_losses is not None and use_inner_triplet_loss:
            for i in range(min(len(inner_outer_layer_weight), len(inner_triplet_losses))):
                aux_loss = aux_loss + inner_outer_layer_weight[i] * inner_triplet_losses[i]
        elif outer_con_losses is not None and use_outer_con_loss:
            for i in range(min(len(inner_outer_layer_weight), len(outer_con_losses))):
                aux_loss = aux_loss + inner_outer_layer_weight[i] * outer_con_losses[i]

    if use_qd_align_loss and qd_align_loss is not None:
        aux_loss = aux_loss + qd_align_weight * qd_align_loss

    return aux_loss


class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 num_emb_list=None,
                 e_dim=64,
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons= None,
                 sk_iters=100,
                 alpha = 1.0,
                 beta = 0.001,
                 n_clusters = 10,
                 sample_strategy = 'all',
                 qd_align = 0.1,
                 trade_off_inner_outer = [1.0, 0.75, 0.25, 0.0],
                 inner_outer_layer_weight = [0.01, 0.01, 0.01, 0.01],
                 use_inner_triplet_loss = True,
                 use_outer_con_loss = True,
                 use_qd_align_loss = True,
        ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.alpha = alpha
        self.beta = beta
        self.n_clusters = n_clusters
        self.sample_strategy = sample_strategy
        self.qd_align = qd_align
        self.trade_off_inner_outer = trade_off_inner_outer
        self.inner_outer_layer_weight = inner_outer_layer_weight
        self.use_inner_triplet_loss = use_inner_triplet_loss
        self.use_outer_con_loss = use_outer_con_loss
        self.use_qd_align_loss = use_qd_align_loss

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)

        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim, beta=self.beta,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,)

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)
    def cos_sim_loss(self, x1, x2, qd_align_w):
        cos_sim = qd_align_w*F.cosine_similarity(x1, x2, dim=-1)
        loss = 1 - cos_sim.mean()
        return loss
    def forward(self, x, q_embs, labels, outer_contrastive_pairs=None, inner_triplet_pairs=None, qd_align_w=None, use_sk=True):
        x = self.encoder(x)
        x_q, rq_loss, outer_con_losses, inner_triplet_losses, indices = self.rq(
            x, labels, outer_contrastive_pairs, inner_triplet_pairs, use_sk=use_sk
        )
        out = self.decoder(x_q)

        qd_align_loss = None
        if self.use_qd_align_loss and qd_align_w is not None:
            q_encode = self.encoder(q_embs)
            qd_align_loss = self.cos_sim_loss(x, q_encode, qd_align_w)
        return out, rq_loss, indices, x_q, outer_con_losses, inner_triplet_losses, qd_align_loss
    
    def vq_initialization(self,x, use_sk=True):
        self.rq.vq_ini(self.encoder(x))

    @torch.no_grad()
    def get_indices(self, xs, labels, use_sk=False):
        x_e = self.encoder(xs)
        _, _,  _, _, indices = self.rq(x_e, labels, use_sk=use_sk)
        return indices
    @torch.no_grad()
    def get_embs(self, xs):
        x_e = self.encoder(xs)
        return x_e
    def compute_loss(self, out, quant_loss, dense_out, outer_con_losses=None, inner_triplet_losses=None, qd_align_loss=None, xs=None):
        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')
        total_loss = loss_recon + self.quant_loss_weight * quant_loss

        aux_loss = _compute_auxiliary_loss(
            outer_con_losses=outer_con_losses,
            inner_triplet_losses=inner_triplet_losses,
            qd_align_loss=qd_align_loss,
            trade_off_inner_outer=self.trade_off_inner_outer,
            inner_outer_layer_weight=self.inner_outer_layer_weight,
            use_outer_con_loss=self.use_outer_con_loss,
            use_inner_triplet_loss=self.use_inner_triplet_loss,
            use_qd_align_loss=self.use_qd_align_loss,
            qd_align_weight=self.qd_align,
        )
        total_loss = total_loss + self.alpha * aux_loss

        return total_loss, loss_recon, quant_loss


class RQKmeans(nn.Module):
    """RQ with K-means codebook update (no gradient update to codebooks). Same interface as RQVAE."""

    def __init__(self,
                 in_dim=768,
                 num_emb_list=None,
                 e_dim=64,
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 kmeans_init=False,
                 kmeans_iters=100,
                 alpha=1.0,
                 beta=0.001,
                 n_clusters=10,
                 sample_strategy='all',
                 qd_align=0.1,
                 trade_off_inner_outer=[1.0, 0.75, 0.25, 0.0],
                 inner_outer_layer_weight=[0.01, 0.01, 0.01, 0.01],
                 use_inner_triplet_loss=True,
                 use_outer_con_loss=True,
                 use_qd_align_loss=True):
        super(RQKmeans, self).__init__()
        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.alpha = alpha
        self.beta = beta
        self.n_clusters = n_clusters
        self.sample_strategy = sample_strategy
        self.qd_align = qd_align
        self.trade_off_inner_outer = trade_off_inner_outer
        self.inner_outer_layer_weight = inner_outer_layer_weight
        self.use_inner_triplet_loss = use_inner_triplet_loss
        self.use_outer_con_loss = use_outer_con_loss
        self.use_qd_align_loss = use_qd_align_loss

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob, bn=self.bn)
        self.rq = ResidualKmeansQuantizer(
            num_emb_list, self.e_dim,
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
        )
        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                 dropout=self.dropout_prob, bn=self.bn)

    def cos_sim_loss(self, x1, x2, qd_align_w):
        cos_sim = qd_align_w * F.cosine_similarity(x1, x2, dim=-1)
        loss = 1 - cos_sim.mean()
        return loss

    def forward(self, x, q_embs, labels, outer_contrastive_pairs=None, inner_triplet_pairs=None, qd_align_w=None, use_sk=True):
        x = self.encoder(x)
        x_q, rq_loss, outer_con_losses, inner_triplet_losses, indices = self.rq(
            x, labels, outer_contrastive_pairs, inner_triplet_pairs, use_sk=use_sk
        )
        out = self.decoder(x_q)
        qd_align_loss = None
        if self.use_qd_align_loss and qd_align_w is not None:
            q_encode = self.encoder(q_embs)
            qd_align_loss = self.cos_sim_loss(x, q_encode, qd_align_w)
        return out, rq_loss, indices, x_q, outer_con_losses, inner_triplet_losses, qd_align_loss

    def vq_initialization(self, x, use_sk=True):
        self.rq.vq_ini(self.encoder(x))

    @torch.no_grad()
    def get_indices(self, xs, labels, use_sk=False):
        x_e = self.encoder(xs)
        _, _, _, _, indices = self.rq(x_e, labels, use_sk=use_sk)
        return indices

    @torch.no_grad()
    def get_embs(self, xs):
        x_e = self.encoder(xs)
        return x_e

    def compute_loss(self, out, quant_loss, dense_out, outer_con_losses=None, inner_triplet_losses=None, qd_align_loss=None, xs=None):
        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')
        total_loss = loss_recon + self.quant_loss_weight * quant_loss
        aux_loss = _compute_auxiliary_loss(
            outer_con_losses=outer_con_losses,
            inner_triplet_losses=inner_triplet_losses,
            qd_align_loss=qd_align_loss,
            trade_off_inner_outer=self.trade_off_inner_outer,
            inner_outer_layer_weight=self.inner_outer_layer_weight,
            use_outer_con_loss=self.use_outer_con_loss,
            use_inner_triplet_loss=self.use_inner_triplet_loss,
            use_qd_align_loss=self.use_qd_align_loss,
            qd_align_weight=self.qd_align,
        )
        total_loss = total_loss + self.alpha * aux_loss
        return total_loss, loss_recon, quant_loss