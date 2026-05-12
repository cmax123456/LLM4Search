#!/bin/bash
source ~/.bashrc  # Try to load conda if needed
eval "$(conda shell.bash hook)" 2>/dev/null
conda activate qwen-3.5

DATASET=esci
DATA_PATH=/data/LLM4Search/data
OUTPUT_DIR=/data/LLM4Search/results
RESULTS_FILE=$OUTPUT_DIR/res_qwen.json
CKPT_PATH=/data/LLM4Search/ckpt/GR/Qwen3_TIGER_ranking_Apr-15-2026_16-42-34/checkpoint-12500

mkdir -p $OUTPUT_DIR

python3 ./GR_train/test_qwen.py \
    --gpu_id 0 \
    --ckpt_path $CKPT_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_datasets game.test,qa.test,random.test \
    --test_batch_size 1 \
    --num_beams 100 \
    --test_prompt_ids 0 \
    --metrics recall@1,recall@10,ndcg@1,ndcg@10,topk_hit@1,topk_hit@3,topk_hit@5,topk_hit@10 \
    --index_file .RQVAE-TIGER.index.json