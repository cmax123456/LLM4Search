#!/bin/bash
source ~/.bashrc  # Try to load conda if needed
eval "$(conda shell.bash hook)" 2>/dev/null
conda activate qwen-3.5

DATASET=esci
# Use default ./ckpt in script, Python code will automatically translate it to /data/LLM4Search/ckpt/GR/...
OUTPUT_DIR=./ckpt
torchrun --nproc_per_node=8 --master_port=2314 /data/LLM4Search/GR_train/finetune_qwen.py \
    --output_dir $OUTPUT_DIR \
    --bf16 \
    --seed 42 \
    --dataset $DATASET \
    --per_device_batch_size 64 \
    --test_batch_size 1 \
    --learning_rate 4e-5 \
    --epochs 50 \
    --index_file .RQVAE-MERGE.index.json \
    --temperature 1.0 \
    --dataset $DATASET \
    --data_path /data/LLM4Search/data_v1 \
    --save_and_eval_strategy steps \
    --save_and_eval_steps 500 \
    --sample_num 200 \
    --num_beams 50 \
    --metrics recall@1,recall@10,ndcg@1,ndcg@10,topk_hit@1,topk_hit@3,topk_hit@5,topk_hit@10 \
    --base_model /mnt/data/user/ici_search/user/caims/qwen3