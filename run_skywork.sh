#!/bin/bash

# 设置显卡 (根据实际情况修改)
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_PATH="/disk/Yitong/models/Skywork/Skywork-o1-Open-Llama-3.1-8B"
HALUEVAL_PATH="./data/qa_data.json" 

# ==========================================
# 任务 1: TruthfulQA
# ==========================================
# echo "Running Skywork-o1 on TruthfulQA..."
# python run_skywork.py \
#     --dataset truthfulqa \
#     --model_path $MODEL_PATH \
#     --output_file "skywork_results/truthfulqa_skywork_results.jsonl" \
#     --max_tokens 1024 \
#     --tensor_parallel_size 4 \
#     --batch_size 16

# ==========================================
# 任务 2: HaluEval
# ==========================================
if [ -f "$HALUEVAL_PATH" ]; then
    echo "Running Skywork-o1 on HaluEval..."
    python run_skywork.py \
        --dataset halueval \
        --data_path $HALUEVAL_PATH \
        --model_path $MODEL_PATH \
        --output_file "skywork_results/halueval_skywork_results.jsonl" \
        --max_tokens 1024 \
        --tensor_parallel_size 4 \
        --batch_size 16
else
    echo "HaluEval data file not found at $HALUEVAL_PATH"
fi