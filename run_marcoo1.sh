#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
MODEL_PATH="/disk/Yitong/models/AIDC-AI/Marco-o1"
HALUEVAL_PATH="./data/qa_data.json" 

# ==========================================
# 任务 1: TruthfulQA
# ==========================================
python run_marcoo1.py \
    --dataset truthfulqa \
    --model_path $MODEL_PATH \
    --output_file "marcoo1_results/truthfulqa_marcoo1_results.jsonl" \
    --max_tokens 1024 \
    --tensor_parallel_size 4 \
    --batch_size 4

# ==========================================
# 任务 2: HaluEval
# ==========================================
# if [ -f "$HALUEVAL_PATH" ]; then
#     python run_marcoo1.py \
#         --dataset halueval \
#         --data_path $HALUEVAL_PATH \
#         --model_path $MODEL_PATH \
#         --output_file "marcoo1_results/halueval_marcoo1_results.jsonl" \
#         --max_tokens 1024 \
#         --tensor_parallel_size 4 \
#         --batch_size 4
# fi