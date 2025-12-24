### FLARE

run_FLARE.py \
生成样例：Qwen2.5-7B-Instruct_truthfulqa_flare_results.csv \
最终答案从<answer></answer>标签中提取

### DRAGIN

run_DRAGIN.py \
生成样例：Qwen2.5-7B-Instruct_truthfulqa_dragin_results.csv \
最终答案从<answer></answer>标签中提取，但是因为模型的问题，现在</answer>后面可能会有多余内容 \
一般速度：6/817 [03:45<10:19:00, 45.80s/it]

### selfRAG

bash run_selfRAG.sh \
生成样例：truthfulqa_selfrag_results.jsonl halueval_selfrag_results.jsonl \
由于模型训练的原因，需要自行进行答案匹配