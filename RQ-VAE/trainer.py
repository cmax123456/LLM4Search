import logging
import json
import numpy as np
import torch
import random
from time import time
from torch import optim
from tqdm import tqdm

import torch.nn.functional as F
from utils import ensure_dir,set_color,get_local_time
import os
from datasets import EmbDataset,ConEmbDataset
from torch.utils.data import DataLoader

class Trainer(object):

    def __init__(self, args, model, writer=None):
        self.args = args
        self.model = model
        self.writer = writer
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = args.ckpt_dir
        saved_model_dir = "{}".format(get_local_time())
        self.ckpt_dir = os.path.join(self.ckpt_dir,saved_model_dir)
        ensure_dir(self.ckpt_dir)
        self.labels = {"0":[],"1":[],"2":[], "3":[],"4":[], "5":[]}
        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.optimizer = self._build_optimizer()
        self.model = self.model.to(self.device)
        self.trained_loss = {"total":[],"rqvae":[],"recon":[],"cf":[]}
        self.valid_collision_rate = {"val":[]}

    def _build_optimizer(self):

        params = self.model.parameters()
        learner =  self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def constrained_km(self, data, n_clusters=10):
        from k_means_constrained import KMeansConstrained 
        x = data
        num_points = len(x)
        if num_points < 2:
            raise ValueError(f"Not enough data points for constrained k-means: {num_points}")

        # Keep all constraints valid for small codebooks.
        # k_means_constrained requires:
        # - n_clusters < num_points
        # - 1 <= size_min <= size_max < num_points
        n_clusters = max(2, min(n_clusters, num_points - 1))
        size_min = max(1, min(num_points // (n_clusters * 2), 10, num_points - 1))
        size_max = max(size_min, min(n_clusters * 6, num_points - 1))

        clf = KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=size_max, max_iter=10, n_init=10,
                                n_jobs=10, verbose=False)
        clf.fit(x)
        t_centers = torch.from_numpy(clf.cluster_centers_)
        t_labels = torch.from_numpy(clf.labels_).tolist()

        return t_centers, t_labels
    
    def vq_init(self):
        self.model.eval()
        original_data = EmbDataset(self.args.data_path)
        init_loader = DataLoader(original_data,num_workers=self.args.num_workers,
                             batch_size=len(original_data), shuffle=True,
                             pin_memory=True)
        print(len(init_loader))
        iter_data = tqdm(
                    init_loader,
                    total=len(init_loader),
                    ncols=100,
                    desc=set_color(f"Initialization of vq","pink"),
                    )
        # Train
        for batch_idx, data in enumerate(iter_data):
            data, emb_idx = data[0], data[1]
            data = data.to(self.device)

            self.model.vq_initialization(data)    
    def _train_epoch(self, train_data, epoch_idx):

        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        total_outer_con_loss = 0
        total_qd_align_loss = 0
        total_inner_triplet_loss = 0
    
        total_quant_loss = 0
        print(len(train_data))
        iter_data = tqdm(
                    train_data,
                    total=len(train_data),
                    ncols=100,
                    desc=set_color(f"Train {epoch_idx}","pink"),
                    )
        embs  = [layer.embedding.weight.cpu().detach().numpy() for layer in self.model.rq.vq_layers]

        for idx, emb in enumerate(embs):
            centers, labels = self.constrained_km(emb)
            self.labels[str(idx)] = labels
        for batch_idx, data in enumerate(iter_data):
            doc_embs, query_embs, outer_contrastive_pairs, inner_triplet_pairs, qd_align_w = data
            doc_embs = doc_embs.to(self.device)
            query_embs = query_embs.to(self.device)
            qd_align_w  = qd_align_w.to(self.device)

            # Respect loss switches: when disabled, do not build corresponding compute graph.
            outer_pairs_arg = outer_contrastive_pairs if getattr(self.model, "use_outer_con_loss", True) else None
            inner_pairs_arg = inner_triplet_pairs if getattr(self.model, "use_inner_triplet_loss", True) else None
            qd_align_arg = qd_align_w if getattr(self.model, "use_qd_align_loss", True) else None
            self.optimizer.zero_grad()
            out, rq_loss, indices, dense_out, outer_con_losses, inner_triplet_losses, qd_align_loss = self.model(
                doc_embs,
                query_embs,
                self.labels,
                outer_pairs_arg,
                inner_pairs_arg,
                qd_align_w=qd_align_arg,
            )
            loss, loss_recon, quant_loss = self.model.compute_loss(out, rq_loss, dense_out, outer_con_losses, inner_triplet_losses, qd_align_loss, xs=doc_embs)
            
            self._check_nan(loss)
            loss.backward()
            self.optimizer.step()
            if hasattr(self.model.rq, "update_all_codebooks"):
                self.model.rq.update_all_codebooks()
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            if outer_con_losses:
                total_outer_con_loss += torch.stack(outer_con_losses).mean().item()
            if qd_align_loss is not None:
                total_qd_align_loss += qd_align_loss.item()
            if inner_triplet_losses:
                total_inner_triplet_loss += torch.stack(inner_triplet_losses).mean().item()
            total_quant_loss += quant_loss.item()
        return total_loss, total_recon_loss, total_outer_con_loss, total_inner_triplet_loss, total_qd_align_loss, quant_loss.item()


    @torch.no_grad()
    def _valid_epoch(self, valid_data):

        self.model.eval()

        iter_data =tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
        indices_set = set()
        l1_set = set()
        l1_l2_set = set()
        l1_l2_l3_set = set()

        num_sample = 0
        embs  = [layer.embedding.weight.cpu().detach().numpy() for layer in self.model.rq.vq_layers]
        for idx, emb in enumerate(embs):
            centers, labels = self.constrained_km(emb)
            self.labels[str(idx)] = labels
            
        # Collect predictions mapped by query to measure semantic sharing
        query_to_docs = valid_data.dataset.query2docs if hasattr(valid_data.dataset, 'query2docs') else {}
        doc_to_query = valid_data.dataset.doc2query if hasattr(valid_data.dataset, 'doc2query') else {}
        
        # We'll store predicted codes by document id
        doc_codes = {}
            
        for batch_idx, data in enumerate(iter_data):

            data, emb_idx = data[0], data[1]
            num_sample += len(data)
            data = data.to(self.device)
            indices = self.model.get_indices(data, self.labels)
            indices = indices.view(-1,indices.shape[-1]).cpu().numpy()
            
            for index, d_id in zip(indices, emb_idx):
                d_id_val = d_id.item()
                code_tuple = tuple([int(_) for _ in index])
                doc_codes[d_id_val] = code_tuple
                
                code = "-".join([str(_) for _ in code_tuple])
                indices_set.add(code)
                
                # Codebook Utilization Rate (CUR) tracker
                if len(index) >= 1:
                    l1_set.add(str(code_tuple[0]))
                if len(index) >= 2:
                    l1_l2_set.add(f"{code_tuple[0]}-{code_tuple[1]}")
                if len(index) >= 3:
                    l1_l2_l3_set.add(f"{code_tuple[0]}-{code_tuple[1]}-{code_tuple[2]}")

        collision_rate = (num_sample - len(indices_set))/num_sample
        
        # Calculate CUR based on standard capacity
        num_emb_list = self.args.num_emb_list
        cur_l1 = len(l1_set) / num_emb_list[0] if len(num_emb_list) >= 1 else 0
        cur_l1_l2 = len(l1_l2_set) / (num_emb_list[0] * num_emb_list[1]) if len(num_emb_list) >= 2 else 0
        cur_l1_l2_l3 = len(l1_l2_l3_set) / (num_emb_list[0] * num_emb_list[1] * num_emb_list[2]) if len(num_emb_list) >= 3 else 0
        
        # Calculate Top-1 Token Ratio and Pairwise Share Rate per layer
        pairwise_share_rate = []
        top1_token_ratio = []
        
        if query_to_docs and len(num_emb_list) > 0:
            for layer_idx in range(len(num_emb_list)):
                total_pairs = 0
                shared_pairs = 0
                total_docs_in_queries = 0
                top1_hits = 0
                
                for query, docs in query_to_docs.items():
                    # For a given query, get the codes of relevant docs (e.g. rel=3 or >0)
                    relevant_docs = [d for d in docs if docs[d] > 0 and d in doc_codes]
                    if len(relevant_docs) == 0:
                        continue
                        
                    layer_tokens = [doc_codes[d][layer_idx] for d in relevant_docs if len(doc_codes[d]) > layer_idx]
                    if len(layer_tokens) == 0:
                        continue
                        
                    # Calculate Top-1 Token Ratio for this query
                    from collections import Counter
                    token_counts = Counter(layer_tokens)
                    top1_hits += token_counts.most_common(1)[0][1]
                    total_docs_in_queries += len(layer_tokens)
                    
                    # Optimized Pairwise Share Rate calculation
                    for count in token_counts.values():
                        if count > 1:
                            shared_pairs += count * (count - 1) // 2
                    
                    n_tokens = len(layer_tokens)
                    total_pairs += n_tokens * (n_tokens - 1) // 2
                                    
                # Aggregate across all queries for this layer
                p_rate = shared_pairs / total_pairs if total_pairs > 0 else 0
                t1_ratio = top1_hits / total_docs_in_queries if total_docs_in_queries > 0 else 0
                pairwise_share_rate.append(p_rate)
                top1_token_ratio.append(t1_ratio)

        return collision_rate, cur_l1, cur_l1_l2, cur_l1_l2_l3, pairwise_share_rate, top1_token_ratio

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):

        ckpt_path = os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_collision_%.4f_model.pth' % (epoch, collision_rate))
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(
            set_color("Saving current", "blue") + f": {ckpt_path}"
        )

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss, total_outer_con_loss, total_inner_triplet_loss, total_qd_align_loss):
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        train_loss_output += set_color("train loss", "blue") + ": %.4f" % loss
        train_loss_output +=", "
        train_loss_output += set_color("reconstruction loss", "blue") + ": %.4f" % recon_loss
        train_loss_output +=", "
        train_loss_output += set_color("outer_con_loss", "blue") + ": %.4f" % total_outer_con_loss
        train_loss_output +=", "
        train_loss_output += set_color("inner_triplet_loss", "blue") + ": %.4f" % total_inner_triplet_loss
        train_loss_output +=", "
        train_loss_output += set_color("qd_align_loss", "blue") + ": %.4f" % total_qd_align_loss
        return train_loss_output + "]"

    def fit(self, data, valid_data):

        cur_eval_step = 0
        self.vq_init()
        for epoch_idx in range(self.epochs):
            # train
            training_start_time = time()
            train_loss, train_recon_loss, total_outer_con_loss, total_inner_triplet_loss, total_qd_align_loss, quant_loss = self._train_epoch(data, epoch_idx)

            training_end_time = time()
            
            # Log training metrics to tensorboard
            if self.writer:
                self.writer.add_scalar("Train/Total_Loss", train_loss, epoch_idx)
                self.writer.add_scalar("Train/Recon_Loss", train_recon_loss, epoch_idx)
                self.writer.add_scalar("Train/Outer_Con_Loss", total_outer_con_loss, epoch_idx)
                self.writer.add_scalar("Train/Inner_Triplet_Loss", total_inner_triplet_loss, epoch_idx)
                self.writer.add_scalar("Train/QD_Align_Loss", total_qd_align_loss, epoch_idx)
                self.writer.add_scalar("Train/Quant_Loss", quant_loss, epoch_idx)
                self.writer.flush()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss, train_recon_loss, total_outer_con_loss, total_inner_triplet_loss, total_qd_align_loss
            )
            self.logger.info(train_loss_output)

            if train_loss < self.best_loss:
                self.best_loss = train_loss

            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                collision_rate, cur_l1, cur_l1_l2, cur_l1_l2_l3, p_share, t1_ratio = self._valid_epoch(valid_data)

                # Log evaluation metrics to tensorboard
                if self.writer:
                    self.writer.add_scalar("Eval/Collision_Rate", collision_rate, epoch_idx)
                    self.writer.add_scalar("Eval/CUR_L1", cur_l1, epoch_idx)
                    self.writer.add_scalar("Eval/CUR_L1_L2", cur_l1_l2, epoch_idx)
                    self.writer.add_scalar("Eval/CUR_L1_L2_L3", cur_l1_l2_l3, epoch_idx)
                    if p_share:
                        for i, val in enumerate(p_share):
                            self.writer.add_scalar(f"Eval/Pairwise_Share_Rate_L{i+1}", val, epoch_idx)
                    if t1_ratio:
                        for i, val in enumerate(t1_ratio):
                            self.writer.add_scalar(f"Eval/Top1_Token_Ratio_L{i+1}", val, epoch_idx)
                    self.writer.flush()

                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    cur_eval_step = 0
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate,
                                          ckpt_file=self.best_collision_ckpt)
                else:
                    cur_eval_step += 1


                valid_end_time = time()
                
                # Format Pairwise Share Rate and Top-1 Token Ratio arrays
                p_share_str = "[" + ", ".join([f"{x:.3f}" for x in p_share]) + "]" if p_share else "N/A"
                t1_ratio_str = "[" + ", ".join([f"{x:.3f}" for x in t1_ratio]) + "]" if t1_ratio else "N/A"
                
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("collision_rate", "blue")
                    + ": %.4f, "
                    + set_color("CUR_l1", "blue")
                    + ": %.4f, "
                    + set_color("CUR_l1_l2", "blue")
                    + ": %.4f, "
                    + set_color("CUR_l1_l2_l3", "blue")
                    + ": %.4f, "
                    + set_color("Pairwise_Share_Rate", "blue")
                    + ": %s, "
                    + set_color("Top1_Token_Ratio", "blue")
                    + ": %s]"
                ) % (epoch_idx, valid_end_time - valid_start_time, collision_rate, cur_l1, cur_l1_l2, cur_l1_l2_l3, p_share_str, t1_ratio_str)

                self.logger.info(valid_score_output)

                if epoch_idx>50:
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate)


        return self.best_loss, self.best_collision_rate




