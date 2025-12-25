#!/bin/bash

# conda create -n etc python=3.9 -y
# conda activate etc
# pip install torch transformers spacy requests datasets tqdm
# python -m spacy download en_core_web_sm

# 运行 TruthfulQA
echo "Running TruthfulQA..."
python run_ETC.py \
    --model_name_or_path "/hub/huggingface/models/meta/Llama-3-8B-Instruct" \
    --dataset "truthfulqa" \


# 运行 HaluEval
# echo "Running HaluEval..."
# python run_ETC.py \
#    --model_name_or_path "/hub/huggingface/models/meta/Llama-3-8B-Instruct" \
#    --dataset "halueval" \
#    --data_path "./data/qa_data.json" \