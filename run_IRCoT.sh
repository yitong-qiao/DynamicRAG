#!/bin/bash

# 配置显卡
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 配置
MODEL_PATH="/hub/huggingface/models/Qwen/Qwen2.5-7B-Instruct" 
HALUEVAL_PATH="./data/qa_data.json"                 
OUTPUT_DIR="IRCoT_results"

mkdir -p $OUTPUT_DIR

# ==========================================
# 任务 1: TruthfulQA
# ==========================================
echo "Running TruthfulQA with IRCoT..."
python run_IRCoT.py \
    --dataset truthfulqa \
    --model_path "$MODEL_PATH" \
    --output_file "$OUTPUT_DIR/truthfulqa_IRCoT_results.jsonl" \
    --max_steps 5 \
    --max_samples 10

# ==========================================
# 任务 2: HaluEval
# ==========================================
if [ -f "$HALUEVAL_PATH" ]; then
    echo "Running HaluEval with IRCoT..."
    python run_IRCoT.py \
        --dataset halueval \
        --data_path "$HALUEVAL_PATH" \
        --model_path "$MODEL_PATH" \
        --output_file "$OUTPUT_DIR/halueval_IRCoT_results.jsonl" \
        --max_steps 5 \
        --max_samples 10
else
    echo "HaluEval data not found, skipping..."
fi