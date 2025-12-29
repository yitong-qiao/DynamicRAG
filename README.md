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

### Skywork-o1

bash run_skywork.sh\
可以设置 batch size 来提升速度 \
可能需要调整一下max_tokens来适应显存 \
生成样例：skywork_results/halueval_skywork_results.jsonl and skywork_results/truthfulqa_skywork_results.jsonl \
因为模型训练格式，有些答案会被包裹在boxed中，而大多数都是最后一句的Final Answer之后；因此可以使用默认最后一句作为conclusive answer来做 EM，可以查看当前的生成样例 \
Skywork/Skywork-o1-Open-Llama-3.1-8B \
https://huggingface.co/Skywork/Skywork-o1-Open-Llama-3.1-8B \

### Marco-o1

bash run_marcoo1.sh\ 
可以设置 batch size 来提升速度 \
可能需要调整一下max_tokens来适应显存 \
生成样例：marcoo1_results/halueval_marcoo1_results.jsonl and marcoo1_results/truthfulqa_marcoo1_results.jsonl \
AIDC-AI/Marco-o1 \
https://huggingface.co/AIDC-AI/Marco-o1 \
最终答案从<answer></answer>标签中提取； 但是某些情况模型可能多输出一些answer标签，建议还是提取</Thought>后方的内容（基本都是最后一句来做EM） \
