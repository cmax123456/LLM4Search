import argparse
import json
import logging
import os
import re
import shutil
import sys
from typing import List

import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen2Config

from modeling_qwen_merge import QwenMerge
from utils import *
from collator_qwen import QwenCollator, QwenTestCollator
from evaluate import get_topk_results, get_metrics_results, get_topk_ranking_results
from generation_trie import Trie


class _TeeStream:
    """Duplicates writes to both a log file and the original stream."""
    def __init__(self, log_fh, original):
        self._log = log_fh
        self._orig = original

    def write(self, data):
        self._orig.write(data)
        self._log.write(data)
        self._log.flush()

    def flush(self):
        self._orig.flush()
        self._log.flush()

    def fileno(self):
        return self._orig.fileno()

    def isatty(self):
        return self._orig.isatty()


class CustomQwenTrainer(transformers.Trainer):
    def __init__(self, *args, candidate_trie=None, custom_tokenizer=None, custom_args=None, test_datasets_dict=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.candidate_trie = candidate_trie
        self.custom_tokenizer = custom_tokenizer
        self.custom_args = custom_args
        self.test_datasets_dict = test_datasets_dict
        self._step_combined_metrics = {}  # global_step -> combined topk_hit@1

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        logger = logging.getLogger(__name__)
        args = self.custom_args
        tokenizer = self.custom_tokenizer
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        final_metrics = {}

        if local_rank == 0:
            model = self.model.module if hasattr(self.model, 'module') else self.model
            model.eval()
            device = next(model.parameters()).device

            orig_padding_side = tokenizer.padding_side
            tokenizer.padding_side = 'left'

            eos_id = tokenizer.eos_token_id

            for test_prefix, test_data in self.test_datasets_dict.items():
                logger.info(f"***** Running Evaluation on {test_prefix} *****")
                collator = QwenTestCollator(args, tokenizer)

                all_items = test_data.get_all_items()
                candidate_trie = Trie(
                    [
                        tokenizer.encode(candidate, add_special_tokens=False)
                        for candidate in all_items
                    ]
                )

                test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
                                         shuffle=False, num_workers=4, pin_memory=True)

                metrics = args.metrics.split(",")
                metrics_results = {}
                total = 0

                def _make_prefix_fn(trie, inp_len):
                    def prefix_allowed_tokens(batch_id, sentence):
                        trie_out = trie.get(sentence.tolist()[inp_len:])
                        return trie_out if trie_out else [eos_id]
                    return prefix_allowed_tokens

                with torch.no_grad():
                    for step, batch in enumerate(tqdm(test_loader, desc=f"Evaluating {test_prefix}")):
                        inputs = batch[0].to(device)
                        targets = batch[1]
                        targets_with_rel = batch[3]
                        total += len(targets)

                        input_length = inputs["input_ids"].shape[1]
                        prefix_allowed_tokens = _make_prefix_fn(candidate_trie, input_length)

                        output = model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=256,
                            prefix_allowed_tokens_fn=prefix_allowed_tokens,
                            num_beams=args.num_beams,
                            num_return_sequences=args.num_beams,
                            output_scores=True,
                            return_dict_in_generate=True,
                            early_stopping=True,
                            pad_token_id=tokenizer.pad_token_id,
                        )

                        output_ids = output["sequences"]
                        generated_ids = output_ids[:, input_length:]
                        output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                        topk_res, generated_results = get_topk_ranking_results(output_text, targets, args.num_beams)
                        batch_metrics_res = get_metrics_results(topk_res, generated_results, targets, targets_with_rel, metrics)

                        for m, res in batch_metrics_res.items():
                            if m not in metrics_results:
                                metrics_results[m] = res
                            else:
                                metrics_results[m] += res

                for m in metrics_results:
                    final_metrics[f"{metric_key_prefix}_{test_prefix}_{m}"] = metrics_results[m] / total

                logger.info("=" * 60)
                logger.info(f"  Evaluation Results on [{test_prefix}]  (total={total})")
                logger.info("=" * 60)
                for m in metrics_results:
                    key = f"{metric_key_prefix}_{test_prefix}_{m}"
                    logger.info(f"  {key:>40s} = {final_metrics[key]:.6f}")
                logger.info("=" * 60)

            tokenizer.padding_side = orig_padding_side

        if world_size > 1:
            import torch.distributed as dist
            metrics_list = [final_metrics]
            dist.broadcast_object_list(metrics_list, src=0)
            final_metrics = metrics_list[0]

        combined_key = f"{metric_key_prefix}_combined_topk_hit@1"
        keys_to_sum = [
            f"{metric_key_prefix}_zhihu.test_topk_hit@1",
            f"{metric_key_prefix}_qa.test_topk_hit@1",
            f"{metric_key_prefix}_game.test_topk_hit@1",
        ]
        final_metrics[combined_key] = sum(final_metrics.get(k, 0.0) for k in keys_to_sum)
        self._step_combined_metrics[self.state.global_step] = final_metrics[combined_key]

        if local_rank == 0:
            logger.info(f"Combined topk_hit@1 = {final_metrics[combined_key]:.6f}")
            logger.info(f"Evaluation Results: {final_metrics}")

        self.log(final_metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, final_metrics)
        return final_metrics

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None):
        """Keep top-4 best checkpoints by combined topk_hit@1 and last-2 recent checkpoints."""
        logger = logging.getLogger(__name__)
        if output_dir is None:
            output_dir = self.args.output_dir

        checkpoints = []
        for name in os.listdir(output_dir):
            path = os.path.join(output_dir, name)
            if not os.path.isdir(path):
                continue
            match = re.match(r"checkpoint-(\d+)$", name)
            if match:
                step = int(match.group(1))
                checkpoints.append((step, path))

        if not checkpoints:
            return

        checkpoints.sort(key=lambda x: x[0])

        last_2_steps = {s for s, _ in checkpoints[-2:]}

        scored = [
            (s, p, self._step_combined_metrics.get(s, float('-inf')))
            for s, p in checkpoints
        ]
        scored.sort(key=lambda x: x[2], reverse=True)
        top_4_steps = {s for s, _, _ in scored[:4]}

        keep_steps = last_2_steps | top_4_steps

        for step, path in checkpoints:
            if step not in keep_steps:
                metric_val = self._step_combined_metrics.get(step)
                metric_str = f"{metric_val:.6f}" if metric_val is not None else "N/A"
                logger.info(
                    f"Deleting checkpoint [{path}] "
                    f"(combined_topk_hit@1={metric_str}, "
                    f"not in top-4 best or last-2 recent)"
                )
                shutil.rmtree(path, ignore_errors=True)


def train(args):
    # Helper to format output directory
    def _get_output_dir_name(args):
        # Get base model name, default to Qwen3 if path contains qwen
        model_name = "Qwen3"
        if "t5" in args.base_model.lower():
            model_name = "T5"
            
        # Get index type (TIGER or MERGE)
        index_name = "MERGE"
        if "tiger" in args.index_file.lower():
            index_name = "TIGER"
            
        # Get loss type
        loss_type = "ranking"
        
        # Get timestamp
        timestamp = get_local_time()
        
        folder_name = f"{model_name}_{index_name}_{loss_type}_{timestamp}"
        return os.path.join("/data/LLM4Search/ckpt/GR", folder_name)

    if args.output_dir == "./ckpt":
        args.output_dir = _get_output_dir_name(args)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    log_file = os.path.join(args.output_dir, "train.log")

    if local_rank == 0:
        _log_fh = open(log_file, "a")
        sys.stdout = _TeeStream(_log_fh, sys.__stdout__)
        sys.stderr = _TeeStream(_log_fh, sys.__stderr__)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Output directory set to: {args.output_dir}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")

    device_map = "auto"
    if local_rank == 0:
        logger.info(f"Arguments: {vars(args)}")

    if ddp:
        device_map = {"": local_rank}
    device = torch.device("cuda", local_rank)
    logger.info(f"Device: {device}")
    logger.info(f"Base model: {args.base_model}")
    
    config = Qwen2Config.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        model_max_length=512,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 151643
        
    args.deepspeed = None

    train_data, valid_data = load_datasets(args)
    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config.vocab_size = len(tokenizer)
    
    if local_rank == 0:
        logger.info(f"Added {add_num} new tokens.")
        logger.info(f"Train data num: {len(train_data)}")
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)
        logger.info(f"Sample train data: {train_data[100]}")
        logger.info(f"Sample valid data: {valid_data[100]}")
    
    # Prepare Trie for evaluation
    all_items = valid_data.get_all_items()
    candidate_trie = Trie(
        [
            tokenizer.encode(candidate, add_special_tokens=False)
            for candidate in all_items
        ]
    )

    test_datasets_str = getattr(args, "test_datasets", "game.test,zhihu.test,qa.test")
    test_prefixes = [x.strip() for x in test_datasets_str.split(",")]
    
    # Pre-load and sample test datasets ONCE before training starts
    test_datasets_dict = {}
    if local_rank == 0:
        logger.info("Pre-loading and sampling test datasets for evaluation...")
    for test_prefix in test_prefixes:
        test_data = load_test_dataset(args, test_prefix=test_prefix)
        test_datasets_dict[test_prefix] = test_data

    collator = QwenCollator(args, tokenizer)
    model = QwenMerge.from_pretrained(args.base_model, config=config, ignore_mismatched_sizes=True, trust_remote_code=True)
    model.set_hyper(args.temperature)
    
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    if local_rank == 0:
        logger.info(model)

    trainer = CustomQwenTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        candidate_trie=candidate_trie,
        custom_tokenizer=tokenizer,
        custom_args=args,
        test_datasets_dict=test_datasets_dict,
        args=transformers.TrainingArguments(
            seed=args.seed,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            logging_steps=args.logging_step,
            optim=args.optim,
            bf16=args.bf16,
            fp16=args.fp16,
            dataloader_num_workers=4,
            eval_strategy=args.save_and_eval_strategy,
            save_strategy=args.save_and_eval_strategy,
            eval_steps=args.save_and_eval_steps,
            save_steps=args.save_and_eval_steps,
            output_dir=args.output_dir,
            logging_dir=os.path.join(args.output_dir, "runs"),
            report_to=["tensorboard"],
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ddp else None,
            eval_delay= 1 if args.save_and_eval_strategy=="epoch" else args.save_and_eval_steps,
        ),
        data_collator=collator,
    )
    model.config.use_cache = False

    trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec Qwen Finetuning')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)  # Add test args to allow setting test_batch_size, num_beams, etc.

    args = parser.parse_args()
    train(args)