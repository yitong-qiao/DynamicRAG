#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
MODEL_PATH="/hub/huggingface/models/Qwen/Qwen2.5-7B-Instruct"
HALUEVAL_PATH="./data/qa_data.json" 

# ==========================================
# 任务 1: TruthfulQA
# ==========================================
python run_CoT_inference.py \
    --dataset truthfulqa \
    --model_path $MODEL_PATH \
    --output_file "CoT_results/truthfulqa_CoT_results.jsonl" \
    --max_tokens 1024 \
    --tensor_parallel_size 4 \
    --batch_size 4

# ==========================================
# 任务 2: HaluEval
# ==========================================
if [ -f "$HALUEVAL_PATH" ]; then
    python run_CoT_inference.py \
        --dataset halueval \
        --data_path $HALUEVAL_PATH \
        --model_path $MODEL_PATH \
        --output_file "CoT_results/halueval_CoT_results.jsonl" \
        --max_tokens 1024 \
        --tensor_parallel_size 4 \
        --batch_size 4
fi