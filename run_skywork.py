import argparse
import json
import os
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ===========================
# Skywork 官方指定的 System Prompt
# ===========================
SKYWORK_SYSTEM_PROMPT = (
    "You are Skywork-o1, a thinking model developed by Skywork AI, "
    "specializing in solving complex problems involving mathematics, coding, and logical reasoning through deep thought. "
    "When faced with a user's request, you first engage in a lengthy and in-depth thinking process to explore possible solutions to the problem. "
    "After completing your thoughts, you then provide a detailed explanation of the solution process in your response."
)

# 自定义的输出格式约束
FORMAT_INSTRUCTION = (
    "First provide your reasoning. "
    "Then provide the final direct answer without detailed illustrations. "
)

def main(args):
    # ===========================
    # 1. 加载数据集
    # ===========================
    all_questions = []
    if args.dataset == "truthfulqa":
        print("Loading TruthfulQA...")
        ds = load_dataset("truthfulqa/truthful_qa", "generation")
        data = ds['validation']
        for item in data:
            all_questions.append(item['question'])
            
    elif args.dataset == "halueval":
        print(f"Loading HaluEval from {args.data_path}...")
        if not args.data_path or not os.path.exists(args.data_path):
            raise FileNotFoundError(f"Data path not found: {args.data_path}")
        with open(args.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    all_questions.append(item['question'])
    
    print(f"Total questions loaded: {len(all_questions)}")
    if args.max_samples is not None:
        all_questions = all_questions[:args.max_samples]
        print(f"Subsampling to {len(all_questions)} samples.")

    # ===========================
    # 2. 检查断点 (Resume Logic)
    # ===========================
    start_index = 0
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf-8') as f:
            existing_lines = f.readlines()
            start_index = len(existing_lines)
            print(f"Found existing output file. Resuming from index {start_index}...")
    
    if start_index >= len(all_questions):
        print("All questions have been processed. Exiting.")
        return

    remaining_questions = all_questions[start_index:]
    
    # ===========================
    # 3. 准备 Tokenizer 和 Model
    # ===========================
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    print(f"Loading model from {args.model_path}...")
    model = LLM(
        model=args.model_path, 
        dtype="auto",
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.80,
        max_model_len=4096 
    )
    
    # 官方参数: temperature=0 (Greedy)
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=args.max_tokens,
        stop_token_ids=None
    )

    # ===========================
    # 4. 分批推理并实时保存
    # ===========================
    batch_size = args.batch_size
    print(f"Starting generation loop with batch size {batch_size}...")

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with tqdm(total=len(remaining_questions), desc="Generating") as pbar:
        for i in range(0, len(remaining_questions), batch_size):
            batch_questions = remaining_questions[i : i + batch_size]
            
            # 构建 Skywork 特定的 Prompt 结构
            batch_prompts = []
            for q in batch_questions:
                conversation = [
                    {
                        "role": "system",
                        "content": SKYWORK_SYSTEM_PROMPT
                    },
                    {
                        "role": "user", 
                        "content": f"{FORMAT_INSTRUCTION}\nQuestion: {q}\nAnswer:"
                    }
                ]
                # 使用 apply_chat_template
                formatted_prompt = tokenizer.apply_chat_template(
                    conversation, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                batch_prompts.append(formatted_prompt)
            
            # 执行推理
            outputs = model.generate(batch_prompts, sampling_params, use_tqdm=False)
            
            # 实时保存
            with open(args.output_file, 'a', encoding='utf-8') as f:
                for q, output in zip(batch_questions, outputs):
                    generated_text = output.outputs[0].text
                    result = {
                        "question": q,
                        "response": generated_text
                    }
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush() 

            pbar.update(len(batch_questions))

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["truthfulqa", "halueval"], help="Dataset name")
    parser.add_argument("--data_path", type=str, default=None, help="Path to local dataset file (for halueval)")
    parser.add_argument("--model_path", type=str, default="/disk/Yitong/models/Skywork/Skywork-o1-Open-Llama-3.1-8B", help="Path to the model")
    parser.add_argument("--output_file", type=str, default="results.jsonl", help="Output file path")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max new tokens")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    
    args = parser.parse_args()
    main(args)