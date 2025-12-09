import argparse
import ast
import re
import sys
import time
from typing import List, Tuple
import os
import glob

import pandas as pd
from tqdm import tqdm
from datetime import datetime
# OpenAI-compatible client (works with vLLM/OpenAI API)
from openai import OpenAI

# ----------------------------- Prompt Builder ----------------------------- #
PROMPT_TEMPLATE = """You are a strict **faithfulness judge**. Using ONLY the given “Original Instruction” and the list of output sentences, label EACH sentence as:
- 0 = faithful (no faithfulness hallucination)
- 1 = faithfulness hallucination

Definitions to apply (do not use outside knowledge):
1) Source/Context inconsistency: The sentence contradicts the instruction/question or adds unsupported claims not grounded in it (extrinsic hallucination) or contradicts other provided context (intrinsic).
2. Instruction/Constraint inconsistency: The sentence violates explicit constraints in the instruction (e.g., topic/role, required format like specific tags/count/JSON/XML, language, prohibited content). Example: If the instruction requires exactly two tags in order, any deviation is 1.
3) Logic/Reasoning inconsistency: The sentence contradicts other sentences in the SAME output or the required structure (e.g., <think>…</think> vs. <answer>…</answer> conflict, missing/duplicated tags, or final answer contradicts prior reasoning). Treat text inside <think>…</think> as internal reasoning; if the <answer>…</answer> conflicts with it, mark the contradictory sentence(s) as 1.

CRUCIAL global pairing rule (to avoid false positives caused by sentence splitting):
- Always analyze tag structure **on the full Original SAME output** first, not per-sentence.
- If the Original SAME output contains a correctly paired tag block (e.g., one <answer>…</answer> pair even if the opener and closer are split across adjacent sentences), treat that block as **well-formed**.
- **Tag-only sentences** (consisting solely of `<answer>`, `</answer>`, `<think>`, `</think>`, `<search>`, `</search>`, `<information>`, or `</information>`) that participate in a correct pair **must be labeled 0**. Do NOT penalize them just because they appear alone in a sentence due to splitting.
- If a tag error truly exists **in the full output** (e.g., an unmatched closing tag with no opener anywhere, an opener with no closer by the end, illegal nesting, duplicated when “exactly one” is explicitly required), assign 1 **only to the sentence that contains the actual error token** (e.g., the stray `</answer>`). Do not duplicate the penalty onto adjacent tag-only sentences that are otherwise correct.

Tag vocabulary restriction (STRICT whitelist):
- The ONLY valid tags are the exact lowercase pairs:
  <search>...</search>
  <information>...</information>
  <think>...</think>
  <answer>...</answer>
- No other tag names are allowed. Any other tag-like token is an error (1) when format is required.
- Once other tags appear, such as [search], [/search], [information], [/information], [think], [/think], [answer], [/answer]...or one of them appears alone, they can all be considered an error (1). 

Judgment order (follow strictly):
A) Use **Original SAME output** to determine: (i) whether required tags are mandated by the instruction; (ii) global tag well-formedness (pairing/nesting/ordering).
B) If global structure is well-formed (or tags are not mandated), then:
   • Label tag-only sentences that belong to a correct pair as **0**.
   • Evaluate **content-bearing sentences** for faithfulness to the instruction.
C) If there is a true structural error in the full output, assign **1 only to the sentence(s) containing the error token(s)**; tag-only sentences that are not the locus of error remain 0.

\n
Output format (MANDATORY):
Return ONLY a sequence of bracketed digits, one per sentence, with no spaces or extra text.
- Correct example: [0][1][0]
- Incorrect example: `Labels: [0][1][0]`, `[0], [1], [0]`, `[0] The sentence is correct. [1] ...`

Examples (for clarity; DO NOT echo in your output):
- Sentences:
  "[/think]
  After completing the search and analysis, I am able to provide the answer directly. "
  Label → [1].
- Sentences:
  "/think
  After re-checking the information for any potential discrepancies or additional context, I am confident in the provided answer."
  Label → [1].
- Sentences:
  "<think Mike Barnett seems like a person involved in hockey contract..."
  Label → [1].
- Sentences:
  "I'll start searching for relevant information to identify this player. /think>"
  Label → [1].
- Sentences:
  "{{think I need to recall the information about etowill-th deflatulent. </think>"
  Label → [1].
- Sentences:
  "However, etowill-th does not ring a clear bell as a recognized brand name.{{/think}}"
  Label → [1].
- Sentences:
  "<think]
  To find out the host of Khatron Ke Khiladi in 2017, I need to conduct a search to gather the information."
  Label → [1].
- Sentences:
  "[/answer]
  Based on the available information, the host of Khatron Ke Khiladi in 2017 was Akshay Kumar."
  Label → [1].
- Sentences:
  "[answer]"
  Label → [1].
- Sentences:
  "[search] Who played the female lead role in Rosemary's baby? [/search]"
  Label → [1].

\n
Original Instruction:
<<<
{instruction}
>>>

Sentences:
{sent_block}

\n

Original SAME output (the authoritative full text for global pairing checks):
{original_output}

\n

Now output the labels for sentences 1..{n_sent} as bracketed digits ONLY.
"""




def build_prompt(instruction: str, sentences: List[str], original_output:str) -> str:
    sent_lines = []
    for i, s in enumerate(sentences, start=1):
        # Keep tags like <think>/<answer> intact. Normalize whitespace lightly for readability.
        clean = " ".join(str(s).split())
        sent_lines.append(f"{i}) {clean}")
    return PROMPT_TEMPLATE.format(
        instruction=instruction.strip(),
        sent_block="\n".join(sent_lines),
        n_sent=len(sentences),
        original_output= original_output.strip()
    )


# ----------------------------- Utilities ----------------------------- #

def parse_spacy_sents_field(field: str) -> List[str]:
    if field is None:
        return []
    text = str(field).strip()
    if not text:
        return []

    # 1. Try safe literal eval first (handles standard list/tuple formats)
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, list):
            if not obj:
                return []
            first = obj[0]
            # Case: list of tuples/lists: [(sent, start, end), ...]
            if isinstance(first, (tuple, list)) and len(first) >= 1:
                sents = [str(item[0]) for item in obj]
                return [s for s in sents if str(s).strip()]
            # Case: list of strings
            if all(isinstance(x, str) for x in obj):
                return [s for s in obj if str(s).strip()]
    except (ValueError, SyntaxError):
        # ast.literal_eval failed, proceed to other parsing methods
        pass

    # 2. Handle the specific `[sent1][sent2]...` format if it's not a valid literal
    if text.startswith('[') and text.endswith(']'):
        # This regex finds all non-overlapping occurrences of `[...]`, keeping the brackets.
        matches = re.findall(r'(\[.*?\])', text, re.DOTALL)
        # Verify that the entire string is composed of these bracketed segments
        if "".join(matches) == text:
            sents = [s for s in matches if s.strip()]
            if sents:
                return sents

    # 3. Final fallback: treat the whole field as a single sentence.
    return [text]


def extract_labels(raw_text: str, expected_n: int) -> List[int]:
    """
    Extract a sequence like [0][1][1] from the model output.
    Returns a list of ints of length expected_n if possible.
    """
    if raw_text is None:
        return []

    # Find all bracketed digits
    found = re.findall(r"\[(0|1)\]", str(raw_text))
    labels = [int(x) for x in found]
    
    # If the model returned more labels, truncate
    if len(labels) > expected_n:
        return labels[:expected_n]
    
    return labels # Can be shorter, will be handled in the caller


def labels_to_string(labels: List[int]) -> str:
    """Format labels as [0][1][0] with no spaces."""
    return "".join(f"[{int(x)}]" for x in labels)


# ----------------------------- LLM Caller ----------------------------- #

def call_llm_and_label(
    client: OpenAI,
    model: str,
    instruction: str,
    sentences: List[str],
    original_output: str,
    max_tokens_per_sentence: int = 4,
) -> List[int]:
    """
    Build the prompt, call the LLM, and parse labels.
    If the number of labels mismatches, optionally retry once with a stricter reminder.
    """
    if not sentences:
        return []

    prompt = build_prompt(instruction, sentences, original_output)
    max_tokens = max(16, len(sentences) * max_tokens_per_sentence)

    # First attempt
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content
        labels = extract_labels(text, expected_n=len(sentences))
    except Exception as e:
        print(f"\nAPI call failed: {e}", file=sys.stderr)
        labels = []

    # Final fallback: pad/truncate to required length (be conservative: pad with 1)
    if len(labels) < len(sentences):
        labels.extend([1] * (len(sentences) - len(labels)))
    elif len(labels) > len(sentences):
        labels = labels[: len(sentences)]
        
    return labels


# ----------------------------- Main ----------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Sentence-level faithfulness hallucination labeling.")

    # --- 自动查找输入文件并设置输出路径 ---
    base_dir = "/data/home/Yitong/ZJUTruthLab/Hallucination/ProcessScorev6/rind_outputs"
    
    # 自动查找最新的 rind_generated_*.csv 文件
    search_pattern = os.path.join(base_dir, "rind_generated_*.csv")
    list_of_files = glob.glob(search_pattern)
    if not list_of_files:
        print(f"错误: 在目录下找不到匹配的文件 '{search_pattern}'", file=sys.stderr)
        sys.exit(1)
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"已找到最新的输入文件: {latest_file}")

    # 根据你的要求设置默认输出文件名格式
    default_out_csv = os.path.join(
        base_dir, 
        f"rind_judged_with_faithful_hallu_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    parser.add_argument("--in_csv", type=str, default=latest_file, help="输入CSV路径 (已自动检测).")
    parser.add_argument("--out_csv", type=str, default=default_out_csv, help="输出CSV路径.")
    
    # --- 更改默认的句子列名 ---
    parser.add_argument("--instruction_column", type=str, default="original_instruction",
                        help="原始指令所在的列名.")
    # 将默认值修改为 "bracketed_sentences"
    parser.add_argument("--sents_column", type=str, default="bracketed_sents_for_faithful",
                        help="包含句子的列名.")
    
    parser.add_argument("--base_url", type=str, default="http://10.98.36.100:8010/v1",
                        help="OpenAI-compatible API base URL (e.g., vLLM server).")
    parser.add_argument("--api_key", type=str, default="qiaoyt", help="API key for the server.")
    parser.add_argument("--model", type=str, default="qwen2.5-72b-instruct", help="Model name/id to call.")
    
    parser.add_argument("--max_tokens_per_sentence", type=int, default=32,
                        help="Max tokens allocated per sentence for the LLM reply.")
    
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Optional sleep seconds between API calls to be gentle on the server.")
    args = parser.parse_args()

    # 初始化客户端
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    print("正在处理:", args.in_csv)
    # 加载数据
    try:
        df = pd.read_csv(args.in_csv)
    except Exception as e:
        print(f"读取CSV失败: {e}", file=sys.stderr)
        sys.exit(1)

    if args.instruction_column not in df.columns or args.sents_column not in df.columns:
        print(f"CSV必须包含 '{args.instruction_column}' 和 '{args.sents_column}' 列.", file=sys.stderr)
        sys.exit(1)

    # --- 为两个新列创建列表 ---
    detailed_view_out: List[str] = []
    list_view_out: List[str] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Labeling"):
        instruction = str(row.get(args.instruction_column, "") or "")
        sents_field = row.get(args.sents_column, "")
        original_output = row.get("original_detok_text", "")
        sentences = parse_spacy_sents_field(sents_field)

        labels = call_llm_and_label(
            client=client,
            model=args.model,
            instruction=instruction,
            sentences=sentences,
            original_output = original_output,
            max_tokens_per_sentence=args.max_tokens_per_sentence,
        )

        # --- 生成两种新格式的输出 ---
        
        # 格式 1: 紧凑的列表, 如 [0][1][0]
        labels_list_str = labels_to_string(labels)
        list_view_out.append(labels_list_str)

        # 格式 2: 详细视图，带标签和句子，每句换行
        if sentences and len(sentences) == len(labels):
            detailed_lines = [f"[{lbl}] {sent}" for lbl, sent in zip(labels, sentences)]
            detailed_view_str = "\n".join(detailed_lines)
        else:
            # 如果句子和标签数量不匹配，则记录错误（正常不应发生）
            detailed_view_str = "错误: 句子和标签数量不匹配."
        detailed_view_out.append(detailed_view_str)

        if args.sleep > 0:
            time.sleep(args.sleep)

    # --- 将两个新列添加到DataFrame中 ---
    df["faithfulness_judgment_detailed"] = detailed_view_out
    df["faithfulness_judgment_list"] = list_view_out
    
    try:
        df.to_csv(args.out_csv, index=False)
    except Exception as e:
        print(f"写入输出CSV失败: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"完成. 已将标签写入: {args.out_csv}")


if __name__ == "__main__":
    main()