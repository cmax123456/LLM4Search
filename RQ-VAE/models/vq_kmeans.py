"""
Vector quantizer with codebook updated by K-means (no gradient update to codebook).
Only commitment loss is used so gradients flow to the encoder only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss


class KmeansVectorQuantizer(nn.Module):
    """VQ layer whose codebook is updated by K-means, not by backprop."""

    def __init__(self, n_e, e_dim, mu=0.25, beta=0.25, kmeans_init=False, kmeans_iters=10):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.mu = mu
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()
        self._last_input = None  # store last batch residual for K-means update

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
        return z_q

    def _kmeans_faiss(self, data, n_clusters):
        x = data.cpu().detach().numpy().astype("float32")
        kmeans = faiss.Kmeans(d=x.shape[1], k=n_clusters, niter=self.kmeans_iters, nredo=3, verbose=False)
        kmeans.train(x)
        t_centers = torch.from_numpy(kmeans.centroids).to(data.device)
        _, labels = kmeans.index.search(x, 1)
        t_labels = labels.flatten().tolist()
        return t_centers, t_labels

    def init_emb(self, data):
        centers, _ = self._kmeans_faiss(data, self.n_e)
        self.embedding.weight.data.copy_(centers)
        self.initted = True

    def update_codebook(self):
        """Run K-means on last stored residual and update codebook. Call after forward (e.g. each batch)."""
        if self._last_input is None:
            return
        data = self._last_input
        self._last_input = None
        if data.shape[0] < self.n_e:
            return
        centers, _ = self._kmeans_faiss(data, self.n_e)
        self.embedding.weight.data.copy_(centers)

    def vq_init(self, x, use_sk=True):
        latent = x.view(-1, self.e_dim)
        if not self.initted:
            self.init_emb(latent)
        d = (
            torch.sum(latent**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()
            - 2 * torch.matmul(latent, self.embedding.weight.t())
        )
        indices = torch.argmin(d, dim=-1)
        x_q = self.embedding(indices).view(x.shape)
        return x_q

    def forward(self, x, label, idx, use_sk=True):
        latent = x.view(-1, self.e_dim)
        if not self.initted and self.training:
            self.init_emb(latent)
        if self.training:
            self._last_input = latent.detach().clone()

        d = (
            torch.sum(latent**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()
            - 2 * torch.matmul(latent, self.embedding.weight.t())
        )
        indices = torch.argmin(d, dim=-1)
        x_q = self.embedding(indices).view(x.shape)

        commitment_loss = F.mse_loss(x_q.detach(), x)
        loss = self.beta * commitment_loss
        x_q = x + (x_q - x).detach()
        indices = indices.view(x.shape[:-1])
        return x_q, loss, indices
