"""
VQ-VAE: Vector Quantized Variational Autoencoder with single codebook.
Same interface as RQVAE for training/encoding (encoder -> single VQ -> decoder).
"""
import torch
from torch import nn
from torch.nn import functional as F

from .layers import MLPLayers
from .vq import VectorQuantizer


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


def _inner_triplet_loss(triplets, features, margin=0.2):
    if not triplets:
        return torch.tensor(0.0, device=features.device)
    triplets = torch.tensor(triplets, dtype=torch.long, device=features.device)
    anchors = features[triplets[:, 0]]
    positives = features[triplets[:, 1]]
    negatives = features[triplets[:, 2]]
    pos_distances = F.pairwise_distance(anchors, positives, p=2)
    neg_distances = F.pairwise_distance(anchors, negatives, p=2)
    losses = F.relu(pos_distances - neg_distances + margin)
    return losses.mean()


def _outer_contrastive_loss(x_q, contrastive_pairs, temperature=0.1):
    device = x_q.device
    if not contrastive_pairs or contrastive_pairs is None:
        return torch.tensor(0.0, device=device)
    idx1 = torch.tensor([p[0] for p in contrastive_pairs], dtype=torch.long, device=device)
    idx2 = torch.tensor([p[1] for p in contrastive_pairs], dtype=torch.long, device=device)
    x_q_norm = F.normalize(x_q, p=2, dim=1)
    K = idx1.size(0)
    chunk_size = 2048
    loss = 0.0
    for i in range(0, K, chunk_size):
        end = min(i + chunk_size, K)
        idx1_chunk = idx1[i:end]
        idx2_chunk = idx2[i:end]
        sim_chunk = torch.matmul(x_q_norm[idx1_chunk], x_q_norm.T) / temperature
        pos_chunk = sim_chunk[torch.arange(end - i, device=device), idx2_chunk].unsqueeze(1)
        mask = torch.ones_like(sim_chunk, dtype=torch.bool)
        mask[torch.arange(end - i, device=device), idx2_chunk] = False
        neg_chunk = sim_chunk.masked_select(mask).view(end - i, -1)
        logits_chunk = torch.cat([pos_chunk, neg_chunk], dim=1)
        labels_chunk = torch.zeros(end - i, dtype=torch.long, device=device)
        loss += F.cross_entropy(logits_chunk, labels_chunk, reduction='sum')
    return loss / K


class VQVAE(nn.Module):
    """VQ-VAE with one vector quantizer. API compatible with RQVAE for trainer/encode/generate_indices."""

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
                 sk_epsilons=None,
                 sk_iters=100,
                 alpha=1.0,
                 beta=0.001,
                 n_clusters=10,
                 sample_strategy='all',
                 qd_align=0.1,
                 trade_off_inner_outer=None,
                 inner_outer_layer_weight=None,
                 use_inner_triplet_loss=True,
                 use_outer_con_loss=True,
                 use_qd_align_loss=True,
                 ):
        super(VQVAE, self).__init__()
        if num_emb_list is None:
            num_emb_list = [256]
        # VQ-VAE uses single codebook: take first (or only) size
        n_e = num_emb_list[0] if isinstance(num_emb_list, (list, tuple)) else num_emb_list
        num_emb_list = [n_e]

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
        sk_eps = (sk_epsilons[0] if sk_epsilons and len(sk_epsilons) > 0 else 0.0)
        sk_it = sk_iters if sk_iters else 50
        self.alpha = alpha
        self.beta = beta
        self.n_clusters = n_clusters
        self.sample_strategy = sample_strategy
        self.qd_align = qd_align
        self.trade_off_inner_outer = trade_off_inner_outer or [1.0]
        self.inner_outer_layer_weight = inner_outer_layer_weight or [0.01]
        self.use_inner_triplet_loss = use_inner_triplet_loss
        self.use_outer_con_loss = use_outer_con_loss
        self.use_qd_align_loss = use_qd_align_loss

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(
            layers=self.encode_layer_dims,
            dropout=self.dropout_prob,
            bn=self.bn,
        )
        self.vq = VectorQuantizer(
            n_e,
            e_dim,
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_epsilon=sk_eps,
            sk_iters=sk_it,
        )
        # Expose as .rq.vq_layers for trainer / encode / generate_indices
        self.rq = _VQWrapper(self.vq)

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(
            layers=self.decode_layer_dims,
            dropout=self.dropout_prob,
            bn=self.bn,
        )

    def cos_sim_loss(self, x1, x2, qd_align_w):
        cos_sim = qd_align_w * F.cosine_similarity(x1, x2, dim=-1)
        loss = 1 - cos_sim.mean()
        return loss

    def forward(self, x, q_embs, labels, outer_contrastive_pairs=None, inner_triplet_pairs=None, qd_align_w=None, use_sk=True):
        x_e = self.encoder(x)
        q_encode = self.encoder(q_embs)
        # Single layer: use labels["0"]
        label0 = labels.get("0", None)
        x_q, vq_loss, indices = self.vq(x_e, label0, 0, use_sk=use_sk)
        # Ensure indices shape (N, 1) for compatibility with RQ
        indices = indices.unsqueeze(-1)
        out = self.decoder(x_q)
        qd_align_loss = self.cos_sim_loss(x_e, q_encode, qd_align_w)

        # Single-layer: one outer_con and one inner_triplet for API compatibility (trainer expects non-empty lists)
        device = x_q.device
        outer_con_losses = [_outer_contrastive_loss(x_q, outer_contrastive_pairs or [])] if self.use_outer_con_loss else [torch.tensor(0.0, device=device)]
        inner_triplet_losses = [_inner_triplet_loss(inner_triplet_pairs or [], x_e)] if self.use_inner_triplet_loss else [torch.tensor(0.0, device=device)]

        return out, vq_loss, indices, x_q, outer_con_losses, inner_triplet_losses, qd_align_loss

    def vq_initialization(self, x, use_sk=True):
        self.vq.vq_init(self.encoder(x), use_sk=use_sk)

    @torch.no_grad()
    def get_indices(self, xs, labels, use_sk=False):
        x_e = self.encoder(xs)
        label0 = labels.get("0", None)
        _, _, indices = self.vq(x_e, label0, 0, use_sk=use_sk)
        return indices.unsqueeze(-1)

    @torch.no_grad()
    def get_embs(self, xs):
        return self.encoder(xs)

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


class _VQWrapper(nn.Module):
    """Wrapper so that model.rq.vq_layers exists for trainer/encode/generate_indices."""

    def __init__(self, vq):
        super().__init__()
        self.vq_layers = nn.ModuleList([vq])
