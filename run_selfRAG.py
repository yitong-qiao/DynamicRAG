import argparse
import json
import requests
import os
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams

# ===========================
# 1. 检索函数
# ===========================
def search(query: str):
    try:
        payload = {
            "queries": [query],
            "topk": 3,
            "return_scores": True
        }
        response = requests.post("http://10.98.36.100:8003/retrieve", json=payload)
        response.raise_for_status()
        results = response.json()['result']
                    
        def _passages2string(retrieval_result):
            format_reference = ''
            for idx, doc_item in enumerate(retrieval_result):
                content = doc_item['document']['contents']
                title = content.split("\n")[0]
                text = "\n".join(content.split("\n")[1:])
                format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
            return format_reference
        return _passages2string(results[0])
    except Exception as e:
        print(f"Error during retrieval for query '{query}': {e}")
        return ""

# ===========================
# 2. Prompt 格式化 (Self-RAG 规范)
# ===========================
def format_prompt(input_text, paragraph=None):
    """
    根据Self-RAG格式构建Prompt。
    如果是Always Retrieve模式，将检索内容放入[Retrieval]<paragraph>...</paragraph>中。
    """
    prompt = "### Instruction:\n{0}\n\n### Response:\n".format(input_text)
    if paragraph is not None and len(paragraph) > 0:
        # Self-RAG 识别 <paragraph> 标签作为检索到的上下文
        prompt += "[Retrieval]<paragraph>{0}</paragraph>".format(paragraph)
    return prompt

def main(args):
    # ===========================
    # 3. 加载数据集
    # ===========================
    questions = []
    if args.dataset == "truthfulqa":
        print("Loading TruthfulQA...")
        ds = load_dataset("truthfulqa/truthful_qa", "generation")
        data = ds['validation']
        for item in data:
            questions.append(item['question'])
            
    elif args.dataset == "halueval":
        print(f"Loading HaluEval from {args.data_path}...")
        if not os.path.exists(args.data_path):
            raise FileNotFoundError(f"Data path not found: {args.data_path}")
            
        with open(args.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    questions.append(item['question'])
    
    print(f"Total questions loaded: {len(questions)}")
    
    # 为了测试方便，可选：只运行前N个
    if args.max_samples is not None:
        questions = questions[:args.max_samples]
        print(f"Subsampling to {len(questions)} samples.")

    # ===========================
    # 4. 批量检索 (Batch Retrieval)
    # ===========================
    print("Starting retrieval...")
    prompts = []
    # 遍历所有问题进行检索并构建Prompt
    # 注意：为了效率，先构建好所有Prompt再送入vllm
    for q in tqdm(questions, desc="Retrieving"):
        retrieved_context = search(q)
        final_prompt = format_prompt(q, retrieved_context)
        prompts.append(final_prompt)

    # ===========================
    # 5. 模型加载与推理 (vllm)
    # ===========================
    print(f"Loading model from {args.model_path}...")
    # 建议使用 float16 或 bfloat16 以节省显存
    model = LLM(model=args.model_path, dtype="half", trust_remote_code=True)
    
    # 配置采样参数
    # skip_special_tokens=False 看Self-RAG生成的 [Relevant], [Utility] 等标签
    sampling_params = SamplingParams(
        temperature=0.0, 
        top_p=1.0, 
        max_tokens=args.max_tokens, 
        skip_special_tokens=False 
    )

    print("Generating responses...")
    outputs = model.generate(prompts, sampling_params)

    # ===========================
    # 6. 保存结果
    # ===========================
    print(f"Saving results to {args.output_file}...")
    results = []
    for q, output in zip(questions, outputs):
        generated_text = output.outputs[0].text
        results.append({
            "question": q,
            "response": generated_text
        })

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["truthfulqa", "halueval"], help="Dataset name")
    parser.add_argument("--data_path", type=str, default=None, help="Path to local dataset file (for halueval)")
    parser.add_argument("--model_path", type=str, default="/disk/Yitong/models/selfrag/selfrag_llama2_7b", help="Path to the model")
    parser.add_argument("--output_file", type=str, default="results.jsonl", help="Output file path")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max new tokens to generate")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for testing")
    
    args = parser.parse_args()
    main(args)