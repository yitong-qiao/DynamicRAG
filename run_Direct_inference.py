import argparse
import json
import os
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

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
    # 2. 检查断点
    # ===========================
    start_index = 0
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf-8') as f:
            # 计算已经跑了多少行
            existing_lines = f.readlines()
            start_index = len(existing_lines)
            print(f"Found existing output file with {start_index} samples. Resuming from index {start_index}...")
    if start_index >= len(all_questions):
        print("All questions have been processed. Exiting.")
        return
    # 截取还没跑的问题
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
        gpu_memory_utilization=0.80
    )
    
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.7, 
        stop_token_ids=None 
    )

    # ===========================
    # 4. 分批推理并实时保存
    # ===========================
    batch_size = args.batch_size 
    print(f"Starting generation loop with batch size {batch_size}...")

    # 确保目录存在
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_steps = (len(remaining_questions) + batch_size - 1) // batch_size
    
    with tqdm(total=len(remaining_questions), desc="Generating") as pbar:
        for i in range(0, len(remaining_questions), batch_size):
            batch_questions = remaining_questions[i : i + batch_size]
            batch_prompts = []
            for q in batch_questions:
                base_prompt = f"Answer the following question by reasoning step-by-step, following the example above. First provide your reasoning. Then provide the final direct answer enclosed inside <answer> and </answer>. After providing the final answer, end your response.\nQuestion: {q}\nAnswer:"
                messages = [
                    {"role": "user", "content": base_prompt}
                ]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                batch_prompts.append(formatted_prompt)
            # 执行推理
            # vLLM 会在这个调用中并行处理这 batch_size 个数据
            outputs = model.generate(batch_prompts, sampling_params, use_tqdm=False)
            # 立即保存当前批次结果
            with open(args.output_file, 'a', encoding='utf-8') as f:
                for q, output in zip(batch_questions, outputs):
                    generated_text = output.outputs[0].text
                    result = {
                        "question": q,
                        "response": generated_text
                    }
                    # 写入一行并立即 flush 缓冲区，确保写入硬盘
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush() 
            pbar.update(len(batch_questions))
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["truthfulqa", "halueval"], help="Dataset name")
    parser.add_argument("--data_path", type=str, default=None, help="Path to local dataset file (for halueval)")
    parser.add_argument("--model_path", type=str, default="/disk/Yitong/models/AIDC-AI/Marco-o1", help="Path to the model")
    parser.add_argument("--output_file", type=str, default="results.jsonl", help="Output file path")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max new tokens")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size. Set to 1 to save after every question (slower).")
    args = parser.parse_args()
    main(args)