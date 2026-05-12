import argparse
import json
import os
import sys
import random
from typing import List

import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen2Config

from modeling_qwen_merge import QwenMerge
from utils import *
from collator_qwen import QwenTestCollator
from evaluate import get_topk_results, get_metrics_results, get_topk_ranking_results
from generation_trie import Trie


def test(args):
    set_seed(args.seed)
    print(vars(args))
    
    config = Qwen2Config.from_pretrained(args.ckpt_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.ckpt_path,
        model_max_length=512,
        trust_remote_code=True
    )
    
    # Must use left padding for decoder-only model generation
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 151643

    train_data, valid_data = load_datasets(args)
    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config.vocab_size = len(tokenizer)

    print("add {} new token.".format(add_num))
    print("data num:", len(train_data))

    device = torch.device(f"cuda:{args.gpu_id}")
    model = QwenMerge.from_pretrained(
        args.ckpt_path,
        config=config,
        ignore_mismatched_sizes=True,
        trust_remote_code=True
    ).to(device)

    prompt_ids = [0]
    metrics = args.metrics.split(",")

    test_datasets_str = getattr(args, "test_datasets", "game.test,random.test,qa.test")
    test_prefixes = [x.strip() for x in test_datasets_str.split(",")]

    cur = datetime.datetime.now()
    cur_time = cur.strftime("%b-%d-%Y_%H-%M-%S")
    
    # Insert timestamp before .json extension if results_file ends with .json
    results_file = args.results_file
    if results_file.endswith(".json"):
        results_file = results_file[:-5] + f"_{cur_time}.json"
    else:
        results_file = results_file + f"_{cur_time}"
        
    print(f"Results will be saved to: {results_file}")

    global_save_data = {
        "test_prompt_ids": args.test_prompt_ids,
        "results_by_dataset": {}
    }

    for test_prefix in test_prefixes:
        print(f"\n======================================================")
        print(f"Testing on dataset: {test_prefix}")
        print(f"======================================================")

        test_data = load_test_dataset(args, test_prefix=test_prefix)
        collator = QwenTestCollator(args, tokenizer)
        all_items = test_data.get_all_items()

        from collections import defaultdict
        sid_to_pids = defaultdict(list)
        for pid, idx in test_data.product_id_to_index.items():
            if str(idx) in test_data.indices:
                sid = "".join(test_data.indices[str(idx)])
                sid_to_pids[sid].append(pid)

        candidate_trie = Trie(
            [
                tokenizer.encode(candidate, add_special_tokens=False)
                for candidate in all_items
            ]
        )

        test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
                                 shuffle=False, num_workers=4, pin_memory=True)

        print("data num:", len(test_data))

        model.eval()

        all_prompt_results = []
        all_samples = []
        
        eos_id = tokenizer.eos_token_id
        def _make_prefix_fn(trie, inp_len):
            def prefix_allowed_tokens(batch_id, sentence):
                trie_out = trie.get(sentence.tolist()[inp_len:])
                return trie_out if trie_out else [eos_id]
            return prefix_allowed_tokens

        with torch.no_grad():
            for prompt_id in prompt_ids:
                test_loader.dataset.set_prompt(prompt_id)
                metrics_results = {}
                total = 0

                for step, batch in enumerate(tqdm(test_loader)):
                    inputs = batch[0].to(device)
                    targets = batch[1]
                    input_texts = batch[2]
                    targets_with_rel = batch[3]
                    total += len(targets)
                    
                    if step == 0:
                        print(f"Sample inputs for {test_prefix}:", inputs)
                    
                    # Length of the original prompt
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
                    
                    # For causal LMs, slice out the generated tokens
                    generated_ids = output_ids[:, input_length:]

                    output_text = tokenizer.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )

                    topk_res, generated_results = get_topk_ranking_results(output_text, targets, args.num_beams)

                    for idx in range(len(input_texts)):
                        ideal_sorted = sorted(targets_with_rel[idx], key=lambda x: x[1], reverse=True)
                        ideal_top10_sids = [item[0] for item in ideal_sorted[:10]]
                        
                        ideal_top10 = []
                        for sid in ideal_top10_sids:
                            ideal_top10.append({
                                "sid": sid,
                                "product_ids": sid_to_pids.get(sid, [])
                            })
                            
                        generated_top10 = []
                        for sid in generated_results[idx][:10]:
                            generated_top10.append({
                                "sid": sid,
                                "product_ids": sid_to_pids.get(sid, [])
                            })
                        
                        all_samples.append({
                            "query": input_texts[idx],
                            "ideal_top10": ideal_top10,
                            "generated_top10": generated_top10
                        })

                    batch_metrics_res = get_metrics_results(topk_res, generated_results, targets, targets_with_rel, metrics)

                    for m, res in batch_metrics_res.items():
                        if m not in metrics_results:
                            metrics_results[m] = res
                        else:
                            metrics_results[m] += res

                for m in metrics_results:
                    metrics_results[m] = metrics_results[m] / total
                all_prompt_results.append(metrics_results)
                print("Prompt {} results: ".format(prompt_id), metrics_results)

        mean_results = {}
        min_results = {}
        max_results = {}

        for m in metrics:
            all_res = [_[m] for _ in all_prompt_results]
            mean_results[m] = sum(all_res)/len(all_res)
            min_results[m] = min(all_res)
            max_results[m] = max(all_res)

        print("Mean results: ", mean_results)
        
        sampled_20 = random.sample(all_samples, min(20, len(all_samples)))
        
        print(f"\n--- Random 20 Examples for {test_prefix} ---")
        for i, sample in enumerate(sampled_20):
            print(f"Example {i+1}:")
            print(f"Query: {sample['query']}")
            print(f"Ideal Top 10: {sample['ideal_top10']}")
            print(f"Generated Top 10: {sample['generated_top10']}")
            print("-" * 40)
            
        global_save_data["results_by_dataset"][test_prefix] = {
            "mean_results": mean_results,
            "min_results": min_results,
            "max_results": max_results,
            "all_prompt_results": all_prompt_results,
            "random_20_samples": sampled_20
        }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(global_save_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec_test_qwen")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()
    test(args)