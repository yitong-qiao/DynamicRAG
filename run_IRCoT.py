import argparse
import json
import os
import torch
import requests
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===========================
# 1. 搜索与工具函数
# ===========================
def search(query: str, endpoint: str = "http://10.98.36.100:8003/retrieve"):
    try:
        payload = {
            "queries": [query],
            "topk": 3, 
            "return_scores": True
        }
        response = requests.post(endpoint, json=payload, timeout=20)
        results = response.json()['result']
        retrieved_docs = []
        if results and len(results) > 0:
            for doc_item in results[0]:
                content = doc_item['document']['contents']
                parts = content.split("\n")
                title = parts[0]
                text = "\n".join(parts[1:])
                retrieved_docs.append({"title": title, "text": text})
        return retrieved_docs
    except Exception as e:
        # print(f"\n[Warning] Search failed for query '{query}': {e}")
        return []

def format_context(docs):
    """将文档列表格式化为字符串"""
    if not docs:
        return "No context found."
    formatted = ""
    for idx, doc in enumerate(docs):
        formatted += f"Wikipedia Title: {doc['title']}\n{doc['text']}\n\n"
    return formatted

# ===========================
# 2. Prompt 模版
# ===========================
HOTPOT_QA_EXEMPLARS = """
Examples:

Question: Jeremy Theobald and Christopher Nolan share what profession?  
Answer: Jeremy Theobald is an actor and producer. Christopher Nolan is a director, producer, and screenwriter. Therefore, they both share the profession of being a producer. So the answer is <answer>producer</answer>.  

Question: In what country was Lost Gravity manufactured?  
Answer: The Lost Gravity (roller coaster) was manufactured by Mack Rides. Mack Rides is a German company. So the answer is <answer>Germany</answer>.
"""

# ===========================
# 3. IRCoT
# ===========================
def run_ircot_one_sample(model, tokenizer, question, max_steps=4, device="cuda"):
    
    # 状态追踪
    current_reasoning = ""  
    retrieved_docs = []     
    seen_titles = set()     
    
    # 初始检索：先用问题搜一次，建立初始上下文
    # print(f"  [Init] Searching for Question: {question}")
    init_docs = search(question)
    for doc in init_docs:
        if doc['title'] not in seen_titles:
            seen_titles.add(doc['title'])
            retrieved_docs.append(doc)
    
    for step in range(max_steps):
        # A. 构建上下文
        context_str = format_context(retrieved_docs)
        
        # B. 构建 Prompt
        base_prompt = (
            f"{HOTPOT_QA_EXEMPLARS}\n\n"
            f"Reference Context:\n{context_str}\n\n"
            f"Task: Answer the question based on the Reference Context. "
            f"Reason step-by-step in a continuous narrative (do not use bullet points). "
            f"Whenever you have sufficient information, you MUST output the final answer enclosed in <answer>...</answer> tags immediately.\n"
            f"Question: {question}\n"
            f"Answer: {current_reasoning}"
        )
        
        # C. Continue Generation”
        messages = [{"role": "user", "content": base_prompt}]
        model_inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        final_prompt = model_inputs + current_reasoning
        
        inputs = tokenizer(final_prompt, return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128, 
                do_sample=False,    
                pad_token_id=tokenizer.eos_token_id,
                tokenizer=tokenizer,
                stop_strings=["\n", "<|im_end|>", "<|endoftext|>"] # 尝试让模型自己在换行处停下
            )
        
        generated_token_ids = outputs[0][input_len:]
        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        
        # 1. 去除左侧空白
        cleaned_text = generated_text.lstrip()
        
        # 2. 提取第一句话/第一个段落
        if not cleaned_text:
            # 如果这一步模型什么都没吐出来 强行 break 避免死循环
            break
        thought = cleaned_text.split('\n')[0].strip()
        if not thought:
            break

        # 更新推理链
        current_reasoning += thought + " "
        
        # D. 检查是否结束
        if "<answer>" in thought or "</answer>" in thought:
            break
            
        # E. 检索 (Retrieval Step)
        print(f"  [Step {step+1}] Reasoning: {thought}")
        
        # 只有当 thought 比较长或者包含实质内容时才去搜，避免搜停用词
        if len(thought) > 5:
            new_docs = search(thought)
            for doc in new_docs:
                if doc['title'] not in seen_titles:
                    seen_titles.add(doc['title'])
                    retrieved_docs.append(doc)
    
    return current_reasoning

# ===========================
# 4. 主流程
# ===========================
def main(args):
    # --- 加载数据集 ---
    all_questions = []
    if args.dataset == "truthfulqa":
        print("Loading TruthfulQA...")
        ds = load_dataset("truthfulqa/truthful_qa", "generation")
        data = ds['validation']
        for item in data:
            all_questions.append(item['question'])
    elif args.dataset == "halueval":
        print(f"Loading HaluEval from {args.data_path}...")
        with open(args.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    all_questions.append(item['question'])
    
    if args.max_samples:
        all_questions = all_questions[:args.max_samples]
    
    print(f"Total questions: {len(all_questions)}")

    start_index = 0
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf-8') as f:
            start_index = len(f.readlines())
            print(f"Resuming from index {start_index}...")
    
    remaining_questions = all_questions[start_index:]
    if not remaining_questions:
        print("Done.")
        return

    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto", 
        torch_dtype="auto",
        trust_remote_code=True
    )
    model.eval()

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_file, 'a', encoding='utf-8') as f:
        for q in tqdm(remaining_questions, desc="IRCoT Inference"):
            try:
                final_response = run_ircot_one_sample(
                    model, tokenizer, q, 
                    max_steps=args.max_steps, 
                    device=model.device
                )
                
                result = {
                    "question": q,
                    "response": final_response,
                    "method": "IRCoT"
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
            except Exception as e:
                print(f"Error processing question: {q}\nError: {e}")
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["truthfulqa", "halueval"])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="ircot_results.jsonl")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=6, help="Max IRCoT reasoning steps") # 稍微增加步数
    args = parser.parse_args()
    main(args)