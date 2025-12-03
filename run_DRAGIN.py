import transformers
import torch
import random
from datasets import load_dataset
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import numpy as np
import spacy
from scipy.special import softmax
import logging
import re
import argparse
import json
import os
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------ RIND 计算配置 ------------------
ALLOWED_POS = {'NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'}

# ★ FIX: 引入与代码2一致的空白与标签正则 -----------------------
_WHITESPACE_MARKERS = {
    "\u2581": " ",   # SentencePiece/Metaspace
    "\u0120": " ",   # Byte BPE leading space
    "\u010A": "\n",  # Byte BPE newline
}
CLOSE_TAG_RE = re.compile(
    r"^\s*(?:</(?:think|answer|search|information)>)\s*$",
    re.IGNORECASE,
)
LEADING_CLOSE_RE = re.compile(
    r"^\s*(?:</(?:think|answer|search|information)>\s*)+",
    re.IGNORECASE,
)
TAG_BOUNDARY_RE = re.compile(
    r"(</(?:think|answer|search|information)>)|(<\s*(?:search|answer|information)\s*>)",
    re.IGNORECASE,
)
SEARCH_TAG_RE = re.compile(r"<\s*search\s*>", re.IGNORECASE)
ANSWER_TAG_RE = re.compile(r"<\s*answer\s*>", re.IGNORECASE)
TERMINAL_MARKERS = ("<|im_end|>", "<|endoftext|>")

# 修改 防止 <think> 被单独当成一个句子
THINK_TAG_ONLY_RE = re.compile(r"^\s*<\s*think\s*>\s*$", re.IGNORECASE)


# 添加参数设置
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/hub/huggingface/models/Qwen/Qwen2.5-7B-Instruct", help="模型路径")
    parser.add_argument("--dataset", type=str, required=True, choices=["truthfulqa", "halueval"], help="数据集名称")
    parser.add_argument("--data_path", type=str, default="data/qa_data.json", help="Halueval数据路径")
    parser.add_argument("--output_file", type=str, default="results.csv", help="输出CSV文件名")
    parser.add_argument("--save_freq", type=int, default=1, help="保存频率")
    # filter相关参数
    parser.add_argument("--filter_use_api", type=str, default="True", help="Filter是否使用API (True/False)")
    parser.add_argument("--api_base_url", type=str, default="http://10.98.36.100:8018/v1")
    parser.add_argument("--api_key", type=str, default="qiaoyt")
    parser.add_argument("--api_model", type=str, default="qwen2.5-72b-instruct")
    return parser.parse_args()

# RINDCalculator
class RINDCalculator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.config = model.config
        
        # 子词合并时用到的空格标记
        if getattr(self.config, 'model_type', '') == 'llama':
            self.space_token = '▁'
        else:
            space_tokens = tokenizer.tokenize(' ')
            self.space_token = space_tokens[0] if space_tokens else " "

        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # spaCy 用于 POS 判断
        self.nlp = spacy.load('en_core_web_sm')
        self.content_pos = ALLOWED_POS
        self.method = "dragin"  # 或 "attn_prob"

    def is_content_word(self, token_str):
        """语义指示器 s_i：非停用词且属于指定 POS 列表时返回 1，否则返回 0"""
        doc = self.nlp(token_str)
        if len(doc) == 0:
            return 0
        tok = doc[0]
        if tok.is_stop or tok.text.lower() in self.nlp.Defaults.stop_words:
            return 0
        return 1 if tok.pos_ in ALLOWED_POS else 0

    def compute_rind_for_generation(self, generation_outputs, generated_tokens_ids, full_input_ids, solver='max'):
        """
        计算生成文本的RIND得分
        generation_outputs: 生成过程的输出（包含scores）
        generated_tokens_ids: 生成的token ID序列
        """
        # 1. 提取生成token序列
        gen_tokens = self.tokenizer.convert_ids_to_tokens(generated_tokens_ids)
        gen_len = len(generated_tokens_ids)
        
        # 2. 复用生成过程的scores计算熵
        scores = generation_outputs.scores  # 元组，每个元素是(1, vocab_size)的tensor
        all_logits = torch.stack(scores, dim=1).squeeze(0).cpu().numpy()  # (gen_len, vocab_size)
        
        entropies = []
        for i in range(gen_len):
            probs = softmax(all_logits[i])
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(entropy)
        
        # 3. 单独前向传播计算注意力（使用完整上下文）
        # input_ids = generated_tokens_ids.unsqueeze(0).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(full_input_ids, output_attentions=True)
        attentions = outputs.attentions
        
        # 4. 获取最后一层注意力
        last_layer_attn = attentions[-1][0]  # (num_heads, seq_len, seq_len)
        # seq_len = last_layer_attn.shape[1]
        gen_len = len(generated_tokens_ids)
        total_len = full_input_ids.shape[1]
        start_idx = total_len - gen_len # 新生成句子的起始位置
        
        # 5. 聚合注意力
        if solver == "max":
            # head_max, _ = torch.max(last_layer_attn, dim=1)  # [num_heads, seq_len]
            # mean_atten = torch.mean(head_max, dim=0)  # [seq_len]
            # 只取新生成部分的列 (columns correspond to the tokens we are evaluating)
            target_attn = last_layer_attn[:, :, start_idx:] # [heads, total_len, gen_len]
            # 为了符合 RIND 定义 (attention form *subsequent* tokens)，
            # 这里其实直接用生成时的 attention 比较复杂，因为标准 forward 是全注意力的（Masked）。
            # 简单处理：在 total_len 维度上取 max，代表该 token 被全文（主要是后面）关注的最大值。
            head_max, _ = torch.max(target_attn, dim=1) # [heads, gen_len]
            mean_atten = torch.mean(head_max, dim=0) # [gen_len]
        # elif solver == "avg":
        #     head_sum = torch.sum(last_layer_attn, dim=1)  # [num_heads, seq_len]
        #     mean_atten = torch.mean(head_sum, dim=0)  # [seq_len]     
        #     for i in range(seq_len):
        #         mean_atten[i] /= (seq_len - i)
        # elif solver == "last_token":
        #     mean_atten = torch.mean(last_layer_attn[:, -1], dim=0)  # [seq_len]
        else:
            raise ValueError(f"Unknown solver: {solver}")
        
        # 6. 子词合并
        spans = []
        for i, t in enumerate(gen_tokens):
            if i == 0 or t.startswith(self.space_token) or generated_tokens_ids[i] == 13 or (i > 0 and gen_tokens[i-1] == '</s>'):
                spans.append([i, i])
            else:
                spans[-1][1] = i
        
        # 7. 计算每个span的RIND
        rind_list = []
        for (start, end) in spans:
            L = end - start + 1
            
            common_prefixes = {'un', 're', 'in', 'im', 'dis', 'non', 'pre', 'mis', 'sub', 'inter', 'trans'}
            common_suffixes = {'ing', 'ed', 'ly', 'ion', 'able', 'ness', 'ment', 'ful', 'less', 'est', 'ous', 'ive', 's', 'es'}

            word = ''.join(gen_tokens[start:end+1]).replace(self.space_token, '')
            punct_count = sum(1 for tok in gen_tokens[start:end+1] if not tok.isalpha() and not tok.isalnum())
            prefix_count = 1 if any(word.lower().startswith(p) for p in common_prefixes) else 0
            suffix_count = 1 if any(word.lower().endswith(s) for s in common_suffixes) else 0
            L_eff = max(1, L - punct_count - prefix_count - suffix_count)
            
            attn_vals = mean_atten[start:end+1].tolist()
            attn_sum = sum(attn_vals)
            if attn_sum > 0:
                attn_vals = [v / attn_sum for v in attn_vals]
            else:
                attn_vals = [0.0] * len(attn_vals)
            max_attn = max(attn_vals) if attn_vals else 0.0
            
            if self.method == "dragin":
                weight_vals = entropies[start:end+1]
            else:
                weight_vals = [1.0] * L
            span_ent = sum(weight_vals) / L
            
            s = self.is_content_word(word)
            rind = max_attn * span_ent * s * L_eff
            
            pos_tag = self.nlp(word)[0].pos_ if len(self.nlp(word)) > 0 else ""
            rind_list.append((word, rind, max_attn, span_ent, L_eff, pos_tag))
            
        return rind_list
    
# BasicGeneratorRIND
class BasicGeneratorRIND:
    _tokenizer = None
    _model = None
    _config = None

    ############################ 此处use_api选择是不是调用外部的接口 ############################
    def __init__(self, model_path=None, use_api: bool = True,
                 api_base_url: str = None, api_key: str = None, api_model: str = None):
        """
        use_api=True  -> 使用 Chat Completions 接口
        use_api=False -> 使用本地 transformers 模型
        """
        self.use_api = bool(use_api)

        if self.use_api:
            # 延迟导入，避免本地模式下对 openai 依赖
            try:
                from openai import OpenAI
            except Exception as e:
                raise ImportError(
                    "缺少 openai 库。请先 `pip install openai`（>=1.0 版本）。"
                ) from e

            self.api_base_url = api_base_url or "http://10.98.36.100:8018/v1"
            self.api_key = api_key or "YOUR_KEY"
            self.api_model = api_model or "qwen2.5-72b-instruct"
            self._OpenAI = OpenAI  # 保存类引用以便后续实例化
            self.client = self._OpenAI(base_url=self.api_base_url, api_key=self.api_key)

            # API 模式下不需要 tokenizer/model/config
            self.tokenizer = None
            self.model = None
            self.config = None
            logger.info(f"[RIND] Using API backend -> model={self.api_model} base_url={self.api_base_url}")
        else:
            # ===== 本地 transformers 模式（保留你原来的缓存逻辑） =====
            if BasicGeneratorRIND._tokenizer is None:
                if not model_path:
                    raise ValueError("本地模式需要提供 model_path。")
                logger.info("Loading tokenizer and model for the first time...")
                BasicGeneratorRIND._tokenizer = AutoTokenizer.from_pretrained(
                    model_path, local_files_only=True
                )
                BasicGeneratorRIND._config = AutoConfig.from_pretrained(
                    model_path,
                    trust_remote_code="falcon" in model_path
                )
                BasicGeneratorRIND._model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    trust_remote_code="falcon" in model_path,
                    output_attentions=True
                )
                BasicGeneratorRIND._model.eval()

            self.tokenizer = BasicGeneratorRIND._tokenizer
            self.model = BasicGeneratorRIND._model
            self.config = BasicGeneratorRIND._config
            logger.info(f"[RIND] Using LOCAL backend -> model_path={model_path}")

    # ---------- helpers ----------
    @staticmethod
    def _parse_bracketed_list(s: str):
        items = re.findall(r"\[(.*?)\]", s, flags=re.DOTALL)
        return [it.strip() for it in items]

    @staticmethod
    def _safe_json_loads(s: str):
        m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        try:
            return json.loads(s)
        except Exception:
            return None

    # ============ 本地模式下的 prompt 渲染 ============
    def _apply_chat_local(self, messages):
        if hasattr(self.tokenizer, "apply_chat_template"):
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
            model_inputs = self.tokenizer(rendered, return_tensors="pt")
            try:
                in_dev = self.model.get_input_embeddings().weight.device
            except Exception:
                in_dev = next(self.model.parameters()).device
            model_inputs = {k: v.to(in_dev) for k, v in model_inputs.items()}
            return model_inputs, rendered
        else:
            chunks = []
            for m in messages:
                role = m["role"].capitalize()
                chunks.append(f"{role}: {m['content']}")
            chunks.append("Assistant:")
            rendered = "\n".join(chunks)
            model_inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
            return model_inputs, rendered

    # ============ 统一的推理入口（根据 use_api 分支） ============
    def _infer_text(self, messages, max_new_tokens: int):
        if self.use_api:
            return self._infer_text_via_api(messages, max_new_tokens)
        else:
            return self._infer_text_via_local(messages, max_new_tokens)

    # --- 本地 transformers 生成 ---
    def _infer_text_via_local(self, messages, max_new_tokens: int):
        model_inputs, rendered = self._apply_chat_local(messages)
        input_len = model_inputs["input_ids"].shape[1]
        outputs = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            output_attentions=False
        )
        gen_text = self.tokenizer.decode(outputs.sequences[0, input_len:], skip_special_tokens=True)
        return gen_text

    # --- Chat Completions API 生成 ---
    def _infer_text_via_api(self, messages, max_new_tokens: int):
        """
        直接把 messages 传给 /chat/completions；与你给的示例一致。
        """
        try:
            resp = self.client.chat.completions.create(
                model=self.api_model,
                messages=messages,
                max_tokens=int(max_new_tokens),
                temperature=0.0,
                top_p=1.0,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.exception("API 调用失败")
            raise

    # ----------------- Stage 1: 仅指代消解 -----------------
    def _build_coref_only_prompt(self, sentences):
        sys = (
            "You are an expert coreference resolver. Do ONE task only: perform STRICT left-to-right coreference rewriting.\n"
            "For EACH sentence i, if there is a third-person antecedent in EARLIER sentences, REPLACE the referring form "
            "with the MOST RECENT and MOST SPECIFIC antecedent NP. Make MINIMAL edits (no added facts). Keep order; NEVER merge.\n"
            "OUTPUT VALID JSON ONLY with schema:\n"
            "{ \"items\": [ {\"index\":1, \"rewritten\":\"...\"}, {\"index\":2, \"rewritten\":\"...\"} ] }"
        )
        rules = (
            "HARD CONSTRAINTS:\n"
            "- Only resolve THIRD-PERSON referential forms (he, she, they, them, their, it, this/that/these/those, "
            "generic definites like 'the drug/city/company' when clearly referring back).\n"
            "- NEVER rewrite FIRST/SECOND person: {I, me, my, mine, we, us, our, ours, you, your, yours}. "
            "These denote speaker/hearer and are NOT coreference to entities in the text.\n"
            "- Do NOT rewrite DUMMY/EXPLETIVE 'it' constructions (e.g., 'It's best to...', 'It is important to...'). "
            "Leave them as-is unless 'it' clearly refers to a prior NP.\n"
            "- Do NOT change the semantic agent of clauses (e.g., 'I do not have...' must remain first-person).\n"
            "- Preserve entities, numbers, units, and relations. Minimal grammar fixes allowed (agreement; expand contractions after substitution)."
        )
        procedure = (
            "PROCEDURE per sentence i:\n"
            "1) Check if the pronoun/demonstrative is THIRD-PERSON and likely refers to a PRIOR NP. "
            "If it is first/second person or dummy 'it', DO NOT substitute.\n"
            "2) If a valid antecedent exists, substitute with that NP; for possessives, use '<ANTECEDENT>\\'s'.\n"
            "3) Fix agreement and expand contractions as needed (e.g., \"It's\" -> \"<ANTECEDENT> is\").\n"
            "4) Ensure no unresolved {it/its/it's/they/them/their/theirs/this/that/these/those/he/him/his/she/her/hers} remain "
            "WHEN an antecedent exists.\n"
            "5) Keep order. One output item per input sentence."
        )
        examples = (
            "EXAMPLE A (positive):\n"
            "Input:\n"
            "[1] The giant panda has distinctive black and white fur.\n"
            "[2] It's native to central China and primarily feeds on bamboo.\n"
            "JSON:\n"
            "{\"items\":[\n"
            " {\"index\":1,\"rewritten\":\"The giant panda has distinctive black and white fur.\"},\n"
            " {\"index\":2,\"rewritten\":\"The giant panda is native to central China and primarily feeds on bamboo.\"}\n"
            "]}\n\n"
            "EXAMPLE B (negative—do NOT substitute 'I' or dummy 'it'):\n"
            "Input:\n"
            "[1] Buxoluric rotacaps are a type of medication, but I don't have specific details on their composition.\n"
            "[2] I do not have detailed information about buxoluric rotacaps.\n"
            "[3] For accurate details, it's best to consult a medical professional.\n"
            "JSON:\n"
            "{\"items\":[\n"
            " {\"index\":1,\"rewritten\":\"Buxoluric rotacaps are a type of medication, but I don't have specific details on their composition.\"},\n"
            " {\"index\":2,\"rewritten\":\"I do not have detailed information about buxoluric rotacaps.\"},\n"
            " {\"index\":3,\"rewritten\":\"For accurate details, it's best to consult a medical professional.\"}\n"
            "]}\n\n"
            "EXAMPLE C (mixed):\n"
            "Input:\n"
            "[1] Bikacin solution is used for bacterial infections.\n"
            "[2] It is applied topically, but I cannot confirm the dosage.\n"
            "JSON:\n"
            "{\"items\":[\n"
            " {\"index\":1,\"rewritten\":\"Bikacin solution is used for bacterial infections.\"},\n"
            " {\"index\":2,\"rewritten\":\"Bikacin solution is applied topically, but I cannot confirm the dosage.\"}\n"
            "]}"
        )
        sent_lines = "\n".join([f"[{i+1}] {s}" for i, s in enumerate(sentences)])
        user = (
            f"{rules}\n\n{procedure}\n\n{examples}\n\n"
            "Input SENTENCES (1-based):\n"
            f"{sent_lines}\n\n"
            "Return ONLY the JSON object. Do not include any extra text."
        )
        return [{"role": "system", "content": sys},
                {"role": "user", "content": user}]

    def resolve_coref(self, sentences, max_new_tokens: int = 1024):
        messages = self._build_coref_only_prompt(sentences)
        print("=== Stage-1 Coref Prompt ===")
        gen_text = self._infer_text(messages, max_new_tokens=max_new_tokens)
        data = self._safe_json_loads(gen_text)
        if not data or "items" not in data or not isinstance(data["items"], list):
            logger.warning("Stage-1 coref: invalid JSON. Raw:\n%s", gen_text)
            return sentences, {"error": "coref_parse_fail", "raw": gen_text}

        rewritten = sentences[:]
        for it in data["items"]:
            try:
                idx = int(it.get("index"))
                text = (it.get("rewritten") or "").strip()
                if 1 <= idx <= len(sentences) and text:
                    rewritten[idx-1] = text
            except Exception:
                continue
        return rewritten, data

    # ----------------- Stage 2: 仅CLAIM筛选 -----------------
    def _build_claim_only_prompt(self, sentences, scores):
        # —— 仅CLAIM筛选；并强制丢弃 <answer>... </answer> 区间的句子 ——
        sys = (
            "You are a precise factual-claim filter. For each sentence, mark keep=true if the sentence asserts "
            "ANY checkable proposition about entities/attributes/events/relations/roles/reputations/typical behaviors; "
            "otherwise keep=false. Do NOT rewrite sentences. Decide EACH sentence independently; this task does NOT deduplicate. "
            "Output JSON ONLY with schema:\n"
            "{ \"items\": [ {\"index\":1, \"keep\":true|false}, ... ] }\n\n"
            "GLOBAL OVERRIDE:\n"
            "- Sentences that are inside an <answer> ... </answer> block MUST be marked keep=false regardless of content.\n"
            "- The input list preserves <think>/<answer> tags; treat any text between <answer> and </answer> (including the boundary lines) as ANSWER.\n"
            "- If a sentence contains the literal substrings '<answer>' or '</answer>', mark keep=false."
        )

        rules_keep = (
            "MUST-KEEP (ANY one suffices; hedges like 'often/typically/known for' still count because they assert frequency/reputation):\n"
            "- Definitional/copular: \"X is/are (a/an/the) Y\"; taxonomy/membership/affiliation; roles/appointments/status.\n"
            "- Composition/materials/ingredients: \"X contains/includes/consists of/is made of Y\".\n"
            "- Location/origin/distribution: \"X is located in/Native to/From Y\".\n"
            "- Temporal/quantitative: dates, counts, measures.\n"
            "- Behavioral/typicality: what X typically/often/usually does or how it is generally characterized.\n"
            "- Reputation/cultural-symbolic association: \"X is known/famous for ...\"; \"X is (often) a symbol of ...\"; "
            "\"X is widely regarded as ...\"; national/organizational mascots/emblems.\n"
            "- Comparative/superlative claims when they make a concrete assertion (e.g., \"X is one of the largest Y in Z\")."
        )

        rules_drop = (
            "MUST-DROP (ONLY if these are the sole content):\n"
            "- Pure meta/capability/disclaimer: \"I don't have details\", \"This requires information\".\n"
            "- Pure advice/commands: \"You should...\", \"It's best to...\".\n"
            "- Purely subjective personal opinions without a checkable proposition (e.g., \"I think X is cute\").\n"
            "- Pure questions/placeholders with no asserted fact."
        )

        tie_break = (
            "MIXED-CONTENT RULE:\n"
            "- If a sentence mixes a keep-worthy factual clause with meta/advice/opinion in the SAME sentence, mark keep=true "
            "because it CONTAINS a factual proposition.\n"
            "OVERRIDE ORDER: The <answer> ... </answer> rule overrides everything else (i.e., still keep=false)."
        )

        notes = (
            "SCOPE & NOTES:\n"
            "- Only evaluate sentences from the <think> ... </think> block for keep=true. Anything from <answer> ... </answer> is out-of-scope and MUST be keep=false.\n"
            "- Do NOT use the provided scores to decide keep/drop; they are reference only.\n"
            "- Do NOT drop a sentence merely because similar content has appeared elsewhere; judge each sentence on its own.\n"
            "- Return ONLY the JSON object; no extra text."
        )

        sent_lines = "\n".join([f"[{i+1}] {s}" for i, s in enumerate(sentences)])
        score_lines = "\n".join([f"[{i+1}] {scores[i]}" for i in range(len(scores))])

        user = (
            f"{rules_keep}\n\n{rules_drop}\n\n{tie_break}\n\n{notes}\n\n"
            "Input SENTENCES (1-based):\n"
            f"{sent_lines}\n\n"
            "Input SCORES (aligned; reference ONLY — do NOT use to decide):\n"
            f"{score_lines}\n\n"
            "Return only the JSON."
        )

        return [{"role": "system", "content": sys},
                {"role": "user", "content": user}]


    def filter_claims(self, sentences_after_coref, scores, max_new_tokens: int = 1024):
        messages = self._build_claim_only_prompt(sentences_after_coref, scores)
        print("=== Stage-2 Claim Prompt ===")
        # print(messages)
        gen_text = self._infer_text(messages, max_new_tokens=max_new_tokens)
        data = self._safe_json_loads(gen_text)
        if not data or "items" not in data or not isinstance(data["items"], list):
            logger.warning("Stage-2 claim: invalid JSON. Raw:\n%s", gen_text)
            return [False]*len(sentences_after_coref), {"error": "claim_parse_fail", "raw": gen_text}

        keep_flags = [False]*len(sentences_after_coref)
        for it in data["items"]:
            try:
                idx = int(it.get("index"))
                keep = bool(it.get("keep"))
                if 1 <= idx <= len(keep_flags):
                    keep_flags[idx-1] = keep
            except Exception:
                continue
        return keep_flags, data

    # ---------- main method (两阶段串联) ----------
    def judge_sentence(self, bracketed_sentences: str, bracketed_scores: str, max_new_tokens: int = 1024):
        sentences = self._parse_bracketed_list(bracketed_sentences)
        score_strs = self._parse_bracketed_list(bracketed_scores)
        try:
            scores = [float(x) for x in score_strs]
        except Exception:
            raise ValueError("Scores must be numeric inside brackets, e.g., [2.0373][1.6461].")

        if len(sentences) != len(scores):
            raise ValueError(f"Length mismatch: {len(sentences)} sentences vs {len(scores)} scores.")

        # Stage 1: Coref
        coref_sentences, coref_json = self.resolve_coref(sentences, max_new_tokens=max_new_tokens)

        # print("##########################")
        # print("=== After Coref ===")
        # print(coref_sentences)
        # print("##########################")

        # Stage 2: Claims
        keep_flags, claim_json = self.filter_claims(coref_sentences, scores, max_new_tokens=max_new_tokens)

        # final_sentences = coref_sentences
        # final_scores = [scores[i] if keep else -1.0 for i, keep in enumerate(keep_flags)]

        # combined_json = {"coref": coref_json, "claims": claim_json}
        # return final_sentences, final_scores, combined_json

        # 删除掉不保留的句子
        final_sents = []
        final_scores = []
        for i, keep in enumerate(keep_flags):
            if keep:
                final_sents.append(coref_sentences[i])
                final_scores.append(scores[i])
        
        combined_json = {"coref": coref_json, "claims": claim_json}
        return final_sents, final_scores, combined_json







# ==========================================
# DRAGIN 核心组件: QFS (Query Formulation based on Self-attention)
# ==========================================
def formulate_query_qfs(model, tokenizer, full_input_ids, trigger_index, top_n=35):
    """
    基于 Self-Attention 构建查询 (QFS)
    full_input_ids: 包含 prompt + generated_so_far 的完整 ids
    trigger_index: 在 full_input_ids 中触发检索的 token 的绝对索引
    """
    # 确保输入格式
    if full_input_ids.dim() == 1:
        full_input_ids = full_input_ids.unsqueeze(0)
    
    # 获取完整的 Attention Map (需要一次 forward pass 来获取该 token 对全局的注意力)
    with torch.no_grad():
        outputs = model(full_input_ids, output_attentions=True)
    
    # 获取最后一层 Attention: [batch, num_heads, seq_len, seq_len]
    # 取 batch 0
    last_layer_attn = outputs.attentions[-1][0] 
    
    # 聚合多头 (通常取平均或最大，DRAGIN 源码中使用 max 或 mean，这里参考 generate.py 的 solver="max" 逻辑)
    # [seq_len, seq_len]
    attn_matrix, _ = torch.max(last_layer_attn, dim=0) 
    
    # 获取触发 token 对之前所有 token 的注意力权重
    # trigger_index 是行，列是 0 到 trigger_index-1
    # 注意：我们要找的是 trigger_token "关注" 谁
    token_attn = attn_matrix[trigger_index, :trigger_index]
    
    # 选出 Top-N
    k = min(top_n, len(token_attn))
    top_indices = torch.topk(token_attn, k=k).indices
    
    # 必须按原始顺序排列，保证语序通顺
    top_indices_sorted = sorted(top_indices.cpu().tolist())
    
    # 解码生成 Query
    selected_ids = full_input_ids[0][top_indices_sorted]
    query_text = tokenizer.decode(selected_ids, skip_special_tokens=True)
    
    # 清洗 query
    query_text = query_text.replace("\n", " ").strip()
    return query_text

# ==========================================
# 搜索函数封装 (适配 DRAGIN 格式)
# ==========================================
def search_dragin(query: str):
    """
    使用你提供的 API 进行检索，并格式化为 DRAGIN 要求的 Prompt 格式
    格式参考 src/generate.py: [1] Doc1 \n [2] Doc2 ...
    """
    try:
        query = query.replace("\n", " ").strip()[:128]
        payload = {
            "queries": [query],
            "topk": 3, # DRAGIN 默认 topk=3
            "return_scores": True
        }
        response = requests.post("http://10.98.36.100:8003/retrieve", json=payload, timeout=10)
        results = response.json()['result']
        
        # 严格参照 DRAGIN 格式构造 retrieved docs string
        docs_text_list = []
        if results and len(results) > 0:
            for idx, doc_item in enumerate(results[0]):
                content = doc_item['document']['contents']
                content = content.replace("\n", " ").strip()
                docs_text_list.append(f"[{idx+1}] {content}")
        
        return docs_text_list
    except Exception as e:
        logging.error(f"Search failed for query '{query}': {e}")
        return []
    


def sanitize_for_bracket(text):
    """防止生成内容中的方括号干扰解析"""
    return text.replace("[", "(").replace("]", ")")

def format_list_custom(items):
    """将列表格式化为 [item1][item2] 的字符串格式，并清洗内容"""
    # 必须清洗，否则如果内容里有 [] 会导致后续解析错位
    return "".join([f"[{sanitize_for_bracket(str(item))}]" for item in items])


class SentenceStopper(transformers.StoppingCriteria):
    """
    用于FLARE：强制模型在生成完一个完整句子（标点符号）后停止
    """
    def __init__(self, tokenizer):
        # 定义句号、问号、感叹号、换行符作为句子结束符
        self.stop_token_ids = []
        for sym in ['.', '?', '!', '\n', '。', '？', '！']:
            self.stop_token_ids.append(tokenizer.encode(sym, add_special_tokens=False)[-1])

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] > 0:
            return input_ids[0, -1].item() in self.stop_token_ids
        return False

# ==========================================
# 句子停止器 & 获取下个句子
# ==========================================

class SentenceStopper(transformers.StoppingCriteria):
    """
    用于FLARE：强制模型在生成完一个完整句子（标点符号）后停止
    """
    def __init__(self, tokenizer):
        self.stop_token_ids = []
        # 定义句号、问号、感叹号、换行符作为句子结束符
        # 注意：不同 tokenizer 的标点 id 可能不同，这里做泛化处理
        for sym in ['.', '?', '!', '\n', '。', '？', '！']:
            encoded = tokenizer.encode(sym, add_special_tokens=False)
            if encoded:
                self.stop_token_ids.append(encoded[-1])

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] > 0:
            return input_ids[0, -1].item() in self.stop_token_ids
        return False
def get_next_sentence_with_scores(model, tokenizer, input_ids, device, max_new_tokens=128):
    stopper = SentenceStopper(tokenizer)
    stopping_criteria = transformers.StoppingCriteriaList([stopper])
    
    # 必须设置 output_scores=True 以便检查置信度
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7, 
        return_dict_in_generate=True,
        output_scores=True
    )
    # 截取新生成的部分
    gen_seq = outputs.sequences[0][input_ids.shape[1]:]
    return gen_seq, outputs.scores




def check_confidence(gen_ids, scores, threshold=0.6):
    """
    检查生成的句子中是否存在低置信度的 token。
    FLARE: active retrieval strategy that only retrieves when LMs generate low-probability tokens.
    """
    # scores 是一个 tuple，每个元素是 (batch_size, vocab_size) 的 tensor
    low_confidence = False
    probs_list = []
    
    for i, token_id in enumerate(gen_ids):
        # 获取当前步的 logits
        step_logits = scores[i][0] # batch index 0
        step_probs = torch.softmax(step_logits, dim=-1)
        token_prob = step_probs[token_id].item()
        probs_list.append(token_prob)
        
        if token_prob < threshold:
            low_confidence = True
            
    return low_confidence, probs_list

# ==========================================
# FLARE Prompt 模板
# ==========================================
HOTPOT_QA_EXEMPLARS = """
Examples:

Question: Jeremy Theobald and Christopher Nolan share what profession?  
Answer: Jeremy Theobald is an actor and producer. Christopher Nolan is a director, producer, and screenwriter. Therefore, they both share the profession of being a producer. So the answer is <answer> producer </answer>.  

Question: What film directed by Brian Patrick Butler was inspired by a film directed by F.W. Murnau?  
Answer: Brian Patrick Butler directed the film The Phantom Hour. The Phantom Hour was inspired by the films such as Nosferatu and The Cabinet of Dr. Caligari. Of these Nosferatu was directed by F.W. Murnau. So the answer is <answer> The Phantom Hour </answer>.  

Question: How many episodes were in the South Korean television series in which Ryu Hye-young played Bo-ra?  
Answer: The South Korean television series in which Ryu Hye-young played Bo-ra is Reply 1988. The number of episodes Reply 1988 has is 20. So the answer is <answer> 20 </answer>.  

Question: Were Lonny and Allure both founded in the 1990s?  
Answer: Lonny (magazine) was founded in 2009. Allure (magazine) was founded in 1991. Thus, of the two, only Allure  was founded in 1990s. So the answer is <answer> no </answer>.  

Question: Vertical Limit stars which actor who also played astronaut Alan Shepard in "The Right Stuff"?  
Answer: The actor who played astronaut Alan Shepard in "The Right Stuff" is Scott Glenn. The movie Vertical Limit  also starred Scott Glenn. So the answer is <answer> Scott Glenn </answer>.  

Question: What was the 2014 population of the city where Lake Wales Medical Center is located?  
Answer: Lake Wales Medical Center is located in the city of Polk County, Florida. The population of Polk County in 2014 was 15,140. So the answer is <answer> 15,140 </answer>.  

Question: Who was born first? Jan de Bont or Raoul Walsh?  
Answer: Jan de Bont was born on 22 October 1943. Raoul Walsh was born on March 11, 1887. Thus, Raoul Walsh was born the first. So the answer is <answer> Raoul Walsh </answer>.  

Question: In what country was Lost Gravity manufactured?  
Answer: The Lost Gravity (roller coaster) was manufactured by Mack Rides. Mack Rides is a German company. So the answer is <answer> Germany </answer>.
"""



# ==========================================
# DRAGIN 主流程
# ==========================================
# ==========================================
# DRAGIN 主流程 (修复死锁与错位版)
# ==========================================
def process_single_question_dragin(question, model, tokenizer, rind_calculator, device, filter_judge):
    # --- 参数设置 ---
    RIND_THRESHOLD = 1.2  
    TOP_N_TOKENS = 35     
    MAX_NEW_TOKENS = 1024 # 和原来的代码保持一致
    
    # Prompt 构造
    base_prompt = f"{HOTPOT_QA_EXEMPLARS}\n\nAnswer the following question by reasoning step-by-step, following the example above. First provide your reasoning. Then provide the final direct answer enclosed inside <answer> and </answer>. After providing the final answer, end your response.\nQuestion: {question}\nAnswer:"
    
    # base_prompt = f"""Answer the given question. First provide your reasoning. Then provide the final direct answer enclosed inside <answer> and </answer>, without detailed illustrations. After providing the final answer, end your response and do not output anything else.\n{HOTPOT_QA_EXEMPLARS}\nQuestion: {question}\nAnswer:"""

    current_context = base_prompt
    generated_text_so_far = "" 
    
    collected_sentences = []
    collected_scores = []
    full_process_log = base_prompt
    
    # 句子缓冲
    current_sentence_buffer = ""

    step = 0
    # 强制停止：防止死循环导致 OOM (设置一个合理的步数上限，比如 50 步)
    max_steps = 50 
    
    while len(tokenizer.encode(generated_text_so_far)) < MAX_NEW_TOKENS and step < max_steps:
        print(f"DEBUG: Current Len: {len(tokenizer.encode(generated_text_so_far))}") 
        step += 1
        
        input_ids = tokenizer.encode(current_context, return_tensors='pt').to(device)

        # 试探性生成
        temp_seq, temp_scores = get_next_sentence_with_scores(model, tokenizer, input_ids, device)
        
        if len(temp_seq) == 0:
            break
        
        # 当前临时生成的句子文本
        temp_text = tokenizer.decode(temp_seq, skip_special_tokens=True)
        # print(f"DEBUG: Temp Text: {temp_text}")
        
        # 计算 RIND
        class MiniOut:
            def __init__(self, scores): self.scores = tuple(scores)
        mini_out = MiniOut(temp_scores)
        full_temp_input_ids = torch.cat([input_ids, temp_seq.unsqueeze(0)], dim=1)
        rind_list = rind_calculator.compute_rind_for_generation(mini_out, temp_seq, full_temp_input_ids, solver='max')
        # rind_list = rind_calculator.compute_rind_for_generation(mini_out, temp_seq, solver='max')
        
        trigger_info = None
        current_token_offset = 0 
        max_score_in_sent = 0.0
        
        for idx, item in enumerate(rind_list):
            word, score, _, _, _, _ = item
            max_score_in_sent = max(max_score_in_sent, score)
            
            word_tokens_len = len(tokenizer.encode(word, add_special_tokens=False))
            
            if score > RIND_THRESHOLD:
                trigger_info = {
                    "word": word,
                    "score": score,
                    "token_start_idx": current_token_offset, 
                    "word_len": word_tokens_len
                }
                break 
            
            current_token_offset += word_tokens_len
            
        if trigger_info:
            # 如果触发位置是 0 (句首)，说明这句话第一个词就有问题。
            # 如果我们截断，就等于什么都没生成。下一轮还是从这里开始，还是这个词，还是触发。
            # 解决方案：如果是句首触发，强制忽略，让模型先把这句话生成完。
            if trigger_info['token_start_idx'] == 0:
                # print(f"DEBUG: Trigger at start of sentence ('{trigger_info['word']}'). Skipping retrieval to enforce progress.")
                trigger_info = None
        
        # === 触发检索 ===
        if trigger_info:
            trigger_start = trigger_info['token_start_idx']
            trigger_end = trigger_start + trigger_info['word_len']
            
            text_before_ids = temp_seq[:trigger_start]
            trigger_ids = temp_seq[trigger_start:trigger_end]
            text_before_trigger = tokenizer.decode(text_before_ids, skip_special_tokens=True)
            
            # 缓存
            current_sentence_buffer += text_before_trigger

            # QFS Input
            t_before = text_before_ids.unsqueeze(0).to(device) if text_before_ids.dim() == 1 else text_before_ids.to(device)
            t_trigger = trigger_ids.unsqueeze(0).to(device) if trigger_ids.dim() == 1 else trigger_ids.to(device)
            if t_before.numel() == 0:
                qfs_input_ids = torch.cat([input_ids, t_trigger], dim=1)
            else:
                qfs_input_ids = torch.cat([input_ids, t_before, t_trigger], dim=1)

            trigger_idx_in_full = qfs_input_ids.shape[1] - 1
            query = formulate_query_qfs(model, tokenizer, qfs_input_ids, trigger_idx_in_full, top_n=TOP_N_TOKENS)
            docs_list = search_dragin(query)
            
            # 1. 更新总生成文本
            generated_text_so_far += text_before_trigger
            
            # 2. 构造检索文档块
            context_block = "\nBelow are the external knowledge references:\n"
            for doc in docs_list:
                context_block += f"{doc}\n"
            
            # 3. 重构 Prompt：Exemplars + 新知识 + 原始问题 + 目前已生成的答案
            # 这样避免了重复堆叠 instruction，逻辑更清晰
            new_prompt = f"{HOTPOT_QA_EXEMPLARS}\n{context_block}\n"
            new_prompt += f"Answer the following question by reasoning step-by-step based on the external knowledge. First provide your reasoning. Then provide the final direct answer enclosed inside <answer> and </answer>. After providing the final answer, end your response.\nQuestion: {question}\nAnswer: {generated_text_so_far}"

            current_context = new_prompt
            
            # 记录日志
            # full_process_log += text_before_trigger
            # full_process_log += f"\n[RIND Trigger: '{trigger_info['word']}' -> Search: '{query}']\n"

        else:
            # === 未触发 或 强制跳过 ===
            # 拼接缓冲区内容和当前生成的句子，形成完整句子
            full_valid_sentence = current_sentence_buffer + temp_text
            # 存入结果列表 (记录的是真正被采纳的完整句子)
            collected_sentences.append(full_valid_sentence.strip())
            print("full_valid_sentence.strip():", full_valid_sentence.strip())
            collected_scores.append(round(max_score_in_sent, 4))
            # 记录完整句子
            full_process_log += full_valid_sentence
            # 清空缓冲区 (因为句子已经结束并记录了)
            current_sentence_buffer = ""

            generated_text_so_far += temp_text
            current_context += temp_text

            if tokenizer.eos_token_id in temp_seq:
                break
            if "</answer>" in full_valid_sentence.lower(): 
                break
            
        
    
    # ★ FIX: 循环结束后，如果缓冲区里还有没拼完的半句话，也需要存下来
    # if current_sentence_buffer:
    #     full_valid_sentence = current_sentence_buffer
    #     collected_sentences.append(full_valid_sentence.strip())
    #     # 对于残句，分数可以给 0 或者最后一次的分数，这里给 0 标记
    #     collected_scores.append(0.0)

    # Filter 逻辑部分
    raw_sents_str = format_list_custom(collected_sentences)
    raw_scores_str = format_list_custom(collected_scores)
    
    # try:
    #     filtered_sents_list, filtered_scores_list, _ = filter_judge.judge_sentence(raw_sents_str, raw_scores_str)
    # except Exception as e:
    #     logger.error(f"Filter failed: {e}")
    #     filtered_sents_list, filtered_scores_list = [], []
        
    # filtered_sents_str = format_list_custom(filtered_sents_list)
    # filtered_scores_str = format_list_custom(filtered_scores_list)

    filtered_sents_str, filtered_scores_str = [], []
    return (full_process_log, 
            raw_scores_str, raw_sents_str, 
            filtered_scores_str, filtered_sents_str)


if __name__ == "__main__":
    # 1. 解析参数
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Model: {args.model_path}...")
    
    # 命名输出文件 (修改后缀为 dragin)
    model_name = os.path.basename(args.model_path.rstrip('/'))
    args.output_file = f"{model_name}_{args.dataset}_dragin_results.csv"
    print(f"Results will be saved to: {args.output_file}")

    # 2. 初始化模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    # 关键：output_attentions=True 必须开启
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="auto", 
        local_files_only=True, attn_implementation="eager",
        output_attentions=True 
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    rind_calculator = RINDCalculator(model, tokenizer)
    # 确保 RINDCalculator 使用 dragin 模式
    rind_calculator.method = "dragin"

    print("Initializing Claim Filter (Judge)...")
    filter_judge = BasicGeneratorRIND(
        use_api=(args.filter_use_api == "True"),
        api_base_url=args.api_base_url,
        api_key=args.api_key,
        api_model=args.api_model
    )
    
    # 3. 加载数据集
    questions = []
    if args.dataset == "truthfulqa":
        print("Loading TruthfulQA...")
        try:
            ds = load_dataset("truthfulqa/truthful_qa", "generation")
            data = ds['validation']
            for item in data:
                questions.append(item['question'])
        except Exception as e:
            print(f"Load TruthfulQA failed: {e}")

    elif args.dataset == "halueval":
        print(f"Loading HaluEval from {args.data_path}...")
        try:
            with open(args.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        questions.append(item['question'])
        except Exception as e:
            print(f"Load HaluEval failed: {e}")
    
    print(f"Total questions to process: {len(questions)}")

    # 4. CSV 初始化
    if not os.path.exists(args.output_file):
        df_header = pd.DataFrame(columns=[
            "Full_Process", 
            "Think_RIND_Scores", 
            "Think_Sentences",
            "Filtered_Think_Scores",   
            "Filtered_Think_Sentences" 
        ])
        df_header.to_csv(args.output_file, index=False)

    # 5. 主循环
    results_buffer = []
    
    for i, question in enumerate(tqdm(questions)):
        try:
            # === 使用 DRAGIN 流程 ===
            full_ctx, raw_scores, raw_sents, filt_scores, filt_sents = process_single_question_dragin(
                question, model, tokenizer, rind_calculator, 
                device, filter_judge
            )
            
            results_buffer.append({
                "Full_Process": full_ctx,
                "Think_RIND_Scores": raw_scores,
                "Think_Sentences": raw_sents,
                "Filtered_Think_Scores": filt_scores,     
                "Filtered_Think_Sentences": filt_sents    
            })
            
            if (i + 1) % args.save_freq == 0:
                df = pd.DataFrame(results_buffer)
                df.to_csv(args.output_file, mode='a', header=False, index=False)
                results_buffer = [] 
                
        except Exception as e:
            logger.error(f"Error processing question index {i}: {e}")
            continue

    if results_buffer:
        df = pd.DataFrame(results_buffer)
        df.to_csv(args.output_file, mode='a', header=False, index=False)

    print("All Done.")