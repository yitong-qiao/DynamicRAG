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
生成样例：selfrag_results/truthfulqa_selfrag_results.jsonl selfrag_results/halueval_selfrag_results.jsonl \
由于模型训练的原因，需要自行进行答案匹配：默认最后一句作为conclusive answer来做 EM

### ETC

使用模型：meta/Llama-3-8B-Instruct, 因为原来的etc github仓库中只提供了该模型的默认配置参数 \
bash run_etc.sh \ 
生成样例：etc_resuls/halueval_etc_results.jsonl and truthfulqa_etc_results.jsonl \
最终答案从<answer></answer>标签中提取 \
注： 对于halueval数据集将max_length设置为128已经足够； 对于truthfulqa则需要设置稍微长一些，如512

