import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import EmbDataset
from models.rqvae import RQVAE, RQKmeans
from models.vqvae import VQVAE
from k_means_constrained import KMeansConstrained
from tqdm import tqdm


def load_model(ckpt_path, device, in_dim):
    checkpoint = torch.load(ckpt_path, map_location=device)
    args = checkpoint["args"]
    model_type = getattr(args, "model_type", "rqvae")
    if model_type == "rq_kmeans":
        model_cls = RQKmeans
    elif model_type == "vqvae":
        model_cls = VQVAE
    else:
        model_cls = RQVAE
    model_kw = dict(
        in_dim=in_dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        beta=args.beta,
        alpha=args.alpha,
        n_clusters=args.n_clusters,
        sample_strategy=args.sample_strategy,
        qd_align=args.qd_align,
        trade_off_inner_outer=args.trade_off_inner_outer,
        inner_outer_layer_weight=args.inner_outer_layer_weight,
    )
    if model_type == "rqvae":
        model_kw["sk_epsilons"] = getattr(args, "sk_epsilons", [0.0, 0.0, 0.0, 0.003])
        model_kw["sk_iters"] = getattr(args, "sk_iters", 50)
    if model_type == "vqvae":
        model_kw["sk_epsilons"] = [getattr(args, "sk_epsilons", [0.0])[0] if getattr(args, "sk_epsilons", None) else 0.0]
        model_kw["sk_iters"] = getattr(args, "sk_iters", 50)
    model = model_cls(**model_kw)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint: epoch={checkpoint['epoch']}, "
          f"best_collision_rate={checkpoint['best_collision_rate']:.4f}")
    return model, args


def build_labels(model, n_clusters=10):
    labels = {}
    for idx, layer in enumerate(model.rq.vq_layers):
        emb = layer.embedding.weight.cpu().detach().numpy()
        size_min = min(len(emb) // (n_clusters * 2), 10)
        clf = KMeansConstrained(
            n_clusters=n_clusters, size_min=size_min,
            size_max=n_clusters * 6, max_iter=10, n_init=10,
            n_jobs=10, verbose=False,
        )
        clf.fit(emb)
        labels[str(idx)] = clf.labels_.tolist()
    return labels


@torch.no_grad()
def encode(model, data_path, labels, device, batch_size=4096, num_workers=4):
    dataset = EmbDataset(data_path)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    all_indices = []
    for data, _ in tqdm(loader, desc="Encoding"):
        data = data.to(device)
        indices = model.get_indices(data, labels)
        all_indices.append(indices.cpu().numpy())

    return np.concatenate(all_indices, axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode documents with trained RQ-VAE")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="Path to embedding .npy file")
    parser.add_argument("--output", type=str, default="doc_codes.npy", help="Output file for codes")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device)
    in_dim = np.load(args.data_path).shape[-1]
    model, train_args = load_model(args.ckpt, device, in_dim)

    print("Building codebook labels...")
    labels = build_labels(model, n_clusters=train_args.n_clusters)

    print("Encoding documents...")
    codes = encode(model, args.data_path, labels, device,
                   batch_size=args.batch_size, num_workers=args.num_workers)

    np.save(args.output, codes)
    print(f"Saved {len(codes)} document codes to {args.output}")
    print(f"Code shape: {codes.shape}  (num_docs x num_quantizers)")
    print(f"Example codes (first 5):\n{codes[:5]}")
