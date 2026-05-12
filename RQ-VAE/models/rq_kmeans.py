"""
Residual quantization with K-means codebook update (no gradient update to codebooks).
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from .vq_kmeans import KmeansVectorQuantizer


class ResidualKmeansQuantizer(nn.Module):
    """Residual quantizer where each layer's codebook is updated by K-means."""

    def __init__(self, n_e_list, e_dim, beta=1, kmeans_init=False, kmeans_iters=100):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.vq_layers = nn.ModuleList([
            KmeansVectorQuantizer(
                n_e,
                e_dim,
                beta=beta,
                kmeans_init=self.kmeans_init,
                kmeans_iters=self.kmeans_iters,
            )
            for n_e in n_e_list
        ])
        print("RQ-Kmeans:", n_e_list)

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def update_all_codebooks(self):
        """Update each layer's codebook by K-means on last forward residuals. Call after each batch."""
        for quantizer in self.vq_layers:
            quantizer.update_codebook()

    def vq_ini(self, x):
        x_q = 0
        residual = x
        for quantizer in self.vq_layers:
            x_res = quantizer.vq_init(residual, use_sk=True)
            residual = residual - x_res
            x_q = x_q + x_res

    def inner_triplet_loss(self, triplets, features, margin=0.2):
        triplets = torch.tensor(triplets, dtype=torch.long)
        anchors = features[triplets[:, 0]]
        positives = features[triplets[:, 1]]
        negatives = features[triplets[:, 2]]
        pos_distances = F.pairwise_distance(anchors, positives, p=2)
        neg_distances = F.pairwise_distance(anchors, negatives, p=2)
        losses = F.relu(pos_distances - neg_distances + margin)
        return losses.mean()

    def outer_contrastive_loss(self, x_q, contrastive_pairs, temperature=0.1):
        device = x_q.device
        if not contrastive_pairs:
            return torch.tensor(0.0, device=device, requires_grad=True)
        idx1 = torch.tensor([pair[0] for pair in contrastive_pairs], dtype=torch.long, device=device)
        idx2 = torch.tensor([pair[1] for pair in contrastive_pairs], dtype=torch.long, device=device)
        x_q_norm = F.normalize(x_q, p=2, dim=1)
        K = idx1.size(0)
        pos_sim = (x_q_norm[idx1] * x_q_norm[idx2]).sum(dim=1, keepdim=True) / temperature
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

    def forward(self, x, labels, outer_contrastive_pairs=None, inner_triplet_pairs=None, use_sk=True):
        all_losses = []
        all_indices = []
        outer_con_losses = []
        inner_triplet_losses = []
        x_q = 0
        residual = x
        for idx, quantizer in enumerate(self.vq_layers):
            label = labels[str(idx)]
            x_res, loss, indices = quantizer(residual, label, idx, use_sk=use_sk)
            if outer_contrastive_pairs is not None:
                outer_con_loss = self.outer_contrastive_loss(residual, outer_contrastive_pairs)
                outer_con_losses.append(outer_con_loss)
            if inner_triplet_pairs is not None:
                inner_triplet_loss = self.inner_triplet_loss(inner_triplet_pairs, residual)
                inner_triplet_losses.append(inner_triplet_loss)
            residual = residual - x_res
            x_q = x_q + x_res
            all_losses.append(loss)
            all_indices.append(indices)
        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        return x_q, mean_losses, outer_con_losses, inner_triplet_losses, all_indices
