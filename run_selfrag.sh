export CUDA_VISIBLE_DEVICES=2,3

python run_selfRAG.py \
    --dataset truthfulqa \
    --model_path "/disk/Yitong/models/selfrag/selfrag_llama2_7b" \
    --output_file "truthfulqa_selfrag_results.jsonl" \
    --max_samples 5


python run_selfRAG.py \
    --dataset halueval \
    --data_path "./data/qa_data.json" \
    --model_path "/disk/Yitong/models/selfrag/selfrag_llama2_7b" \
    --output_file "halueval_selfrag_results.jsonl" \
    --max_samples 5