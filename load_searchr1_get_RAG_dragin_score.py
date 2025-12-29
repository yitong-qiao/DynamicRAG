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

# 配置日志
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
    parser.add_argument("--model_path", type=str, default="/disk/Yitong/models/v05_step_200", help="模型路径")
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

def format_list_custom(items):
    """将列表格式化为 [item1][item2] 的字符串格式"""
    return "".join([f"[{str(item)}]" for item in items])


# 用来进行指代消解 和 claim过滤的配置

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

    def resolve_coref(self, sentences, max_new_tokens: int = 512):
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


    def filter_claims(self, sentences_after_coref, scores, max_new_tokens: int = 512):
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
    def judge_sentence(self, bracketed_sentences: str, bracketed_scores: str, max_new_tokens: int = 512):
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


# ★ FIX: 与代码2一致的规范化与offset构建 -------------------------
def _normalize_piece(piece: str) -> str:
    for k, v in _WHITESPACE_MARKERS.items():
        piece = piece.replace(k, v)
    piece = re.sub(r"[ \t]+", " ", piece)
    return piece if piece.strip() == "" else piece.strip()

def build_offsets_from_ids(tokenizer, ids):
    """从生成的 token ids 反推文本与字符偏移，避免重分词错位。"""
    if ids is None:
        return "", []
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    # 过滤非法id
    ids = [int(t) for t in ids if t is not None and t != -100]
    if not ids:
        return "", []

    pieces = tokenizer.convert_ids_to_tokens(ids) or []
    pieces = [p for p in pieces if p is not None]
    if not pieces:
        return "", []

    resp_text = tokenizer.convert_tokens_to_string(pieces)
    offsets = []
    cursor = 0
    for tok in pieces:
        seg = tokenizer.convert_tokens_to_string([tok])
        seg = _normalize_piece(seg)
        idx = resp_text.find(seg, cursor)
        if idx == -1:
            idx = cursor
        start = idx
        end = start + len(seg)
        offsets.append((start, end))
        cursor = end
    return resp_text, offsets
# ---------------------------------------------------

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

    def compute_rind_for_generation(self, generation_outputs, generated_tokens_ids, solver='max'):
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
        
        # 3. 单独前向传播计算注意力（仅针对生成部分）
        input_ids = generated_tokens_ids.unsqueeze(0).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(input_ids, output_attentions=True)
        attentions = outputs.attentions
        
        # 4. 获取最后一层注意力
        last_layer_attn = attentions[-1][0]  # (num_heads, seq_len, seq_len)
        seq_len = last_layer_attn.shape[1]
        
        # 5. 聚合注意力
        if solver == "max":
            head_max, _ = torch.max(last_layer_attn, dim=1)  # [num_heads, seq_len]
            mean_atten = torch.mean(head_max, dim=0)  # [seq_len]
        elif solver == "avg":
            head_sum = torch.sum(last_layer_attn, dim=1)  # [num_heads, seq_len]
            mean_atten = torch.mean(head_sum, dim=0)  # [seq_len]     
            for i in range(seq_len):
                mean_atten[i] /= (seq_len - i)
        elif solver == "last_token":
            mean_atten = torch.mean(last_layer_attn[:, -1], dim=0)  # [seq_len]
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


# 定义自定义停止条件
class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False

def get_query(text):
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None

def search(query: str):
    payload = {
            "queries": [query],
            "topk": 3,
            "return_scores": True
        }
    results = requests.post("http://10.98.36.100:8003/retrieve", json=payload).json()['result']
                
    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])


def process_single_question(question, model, tokenizer, rind_calculator, stopping_criteria, curr_eos, curr_search_template, device, filter_judge):
    # 1. 准备 Prompt (保持原逻辑)
    question = question.strip()
    if question[-1] != '?':
        question += '?'
    
    prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""


    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    
    # 这里开始 记录全文
    full_context = prompt
    cnt = 0
    max_turns = 15  # 防止死循环 设置一个最大轮数
    
    # 用于收集 Thinking 过程的数据
    collected_think_sentences = []
    collected_think_scores = []

    # 初始化 context 收集列表和当前状态
    collected_think_contexts = []
    current_search_context = "Internal Knowledge (No Search)"
    
    # 2. 循环生成 (逻辑同原代码，但去掉了 print)
    while cnt < max_turns:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)
        
        outputs = model.generate(
            input_ids, attention_mask=attention_mask, max_new_tokens=1024,
            stopping_criteria=stopping_criteria, pad_token_id=tokenizer.eos_token_id,
            do_sample=True, temperature=0.7, return_dict_in_generate=True, output_scores=True
        )

        generated_tokens = outputs.sequences[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        
        
        ####################....
        # rind_scores = rind_calculator.compute_rind_for_generation(
        #         outputs, 
        #         generated_tokens,
        #         solver='max'
        #     )
        
        # print(f"\n=== Generated Text ===")
        # print(output_text)
        
        # print(f"\n=== RIND Scores ===")
        # print(f"{'Word':<15}{'RIND':<10}{'MaxAttn':<10}{'AvgEnt':<10}{'EffLen':<10}{'POS':<8}")
        # for word, rind, attn, ent, tok_num, pos in rind_scores:
        #     print(f"{word:<15}{rind:<10.4f}{attn:<10.4f}{ent:<10.4f}{tok_num:<10}{pos}")
        # print("="*50 + "\n")
        ########################......


        full_context += output_text # 累积全文

        # --- RIND 计算与分句逻辑 (复用你原来的逻辑，但做关键修改) ---
        resp_text, offsets = build_offsets_from_ids(tokenizer, generated_tokens)
        
        # ★ FIX: 按标签边界细分句子段（而非简单字符串分句）
        doc = rind_calculator.nlp(resp_text)
        raw_sents = [
            (span.text, span.start_char, span.end_char)
            for span in doc.sents
            if span.text.strip()
        ]

        # ★ FIX: 在每个句子内部按 TAG_BOUNDARY_RE 切分，并处理句首连续关闭标签
        sentences = []
        for text, s, e in raw_sents:
            parts = []
            cur = 0
            has_match = False
            for m in TAG_BOUNDARY_RE.finditer(text):
                has_match = True
                if m.start() > cur:
                    parts.append((text[cur:m.start()], s + cur, s + m.start()))
                parts.append((m.group(0), s + m.start(), s + m.end()))
                cur = m.end()
            if cur < len(text):
                parts.append((text[cur:], s + cur, e))

            if not has_match:
                m = LEADING_CLOSE_RE.match(text)
                if m:
                    close_part = text[: m.end()]
                    close_end = s + len(close_part)
                    if sentences:
                        prev_text, prev_s, prev_e = sentences[-1]
                        sentences[-1] = (prev_text + close_part, prev_s, close_end)
                    else:
                        sentences.append((close_part, s, close_end))
                    if m.end() < len(text):
                        sentences.append((text[m.end():], close_end, e))
                    continue

            for seg_text, seg_s, seg_e in parts:
                if not seg_text.strip():
                    continue
                if CLOSE_TAG_RE.fullmatch(seg_text):
                    if sentences:
                        prev_text, prev_s, prev_e = sentences[-1]
                        sentences[-1] = (prev_text + seg_text, prev_s, seg_e)
                    else:
                        sentences.append((seg_text, seg_s, seg_e))
                else:
                    sentences.append((seg_text, seg_s, seg_e))

        # 生成 skip_spans (Search/Info/Answer 区块)
        skip_spans = []
        for tag in ("search", "information", "answer"):
            for m in re.finditer(fr"<{tag}>(.*?)</{tag}>", resp_text, re.DOTALL):
                skip_spans.append((m.start(), m.end()))
        skip_spans.sort()
        merged = []
        for s, e in skip_spans:
            if not merged or s > merged[-1][1]:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)
        skip_spans = merged

        L = len(generated_tokens)
        # 遍历句子计算 RIND
        for i, (sent_text, start_pos, end_pos) in enumerate(sentences):
            sent = sent_text
            
            # (1) 基础跳过逻辑
            if (any(f"<{tag}" in sent for tag in ("search", "information", "answer"))
                or any(s <= start_pos < e for s, e in skip_spans) # 在跳过区域内
                or any(marker in sent for marker in TERMINAL_MARKERS)):
                continue
            
            # ★★★★★ 修复单独 <think> 成句的问题 ★★★★★
            if THINK_TAG_ONLY_RE.fullmatch(sent):
                continue

            token_idxs = [idx for idx, (s, e) in enumerate(offsets) if 0 <= idx < L and s >= start_pos and e <= end_pos]
            if not token_idxs:
                # print(f"No tokens for sentence: {sent} -> reward 0")
                continue

            start_idx, end_idx = token_idxs[0], token_idxs[-1]
            start_idx = max(0, min(start_idx, L - 1))
            end_idx = max(0, min(end_idx, L - 1))
            if end_idx < start_idx:
                continue

            # 取出对应的 token_ids 和 scores（保持你原来的MiniOut调用方式）
            sent_tok_ids = generated_tokens[start_idx: end_idx + 1]
            mini_scores   = outputs.scores[start_idx: end_idx + 1]

            class MiniOut:
                def __init__(self, scores): self.scores = tuple(scores)
            mini_out = MiniOut(mini_scores)

            # 计算 RIND 并求 M
            rind_list = rind_calculator.compute_rind_for_generation(
                mini_out,
                sent_tok_ids,
                solver='max'
            )
            M = max((r for _, r, *_ in rind_list), default=0.0)

            j = i + 1
            while j < len(sentences) and CLOSE_TAG_RE.fullmatch(sentences[j][0]):
                j += 1
            
            # 如果不是触发 Search 或 Answer，则认为是思考过程
            is_search = j < len(sentences) and SEARCH_TAG_RE.search(sentences[j][0])
            is_answer = j < len(sentences) and ANSWER_TAG_RE.search(sentences[j][0])
            
            # ★★★ 收集数据 ★★★
            # 只要不是搜索指令本身(上面已跳过)且不是单纯的标签，我们就存下来
            # 如果你想严格只存 "CONTINUE_THINK"，可以加 if not is_search and not is_answer:
            collected_think_sentences.append(sent.strip())
            collected_think_scores.append(round(M, 4)) # 保留4位小数

            # [新增] 记录当前这句话对应的上下文
            collected_think_contexts.append(current_search_context)

        # 3. 搜索与循环控制 (保持原逻辑)
        if outputs.sequences[0][-1].item() in curr_eos:
            break
        
        tmp_query = get_query(tokenizer.decode(outputs.sequences[0], skip_special_tokens=True))
        # print("tmp_query:", tmp_query)

        if tmp_query:
            try:
                search_results = search(tmp_query)
                # print(f"Search Query: {tmp_query}\nSearch Results: {search_results}\n")
            except Exception:
                search_results = ""
            
            # [新增] 更新下一轮生成使用的上下文状态
            clean_res = search_results.replace("\n", " ").strip()
            current_search_context = f"[Query: {tmp_query}] {clean_res}"

        else:
            search_results = ''
        
        search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
        prompt += search_text
        full_context += search_text
        cnt += 1

    # 下方是新增的过滤逻辑 ##########################################
    # 1. 将原始 list 格式化为 [xx][xx] 字符串，供 judge_sentence 解析
    raw_sents_str = format_list_custom(collected_think_sentences)
    raw_scores_str = format_list_custom(collected_think_scores)
    # 2. 调用 Judge 进行过滤 (指代消解 + Claim Filter)
    try:
        # judge_sentence 内部已经做了解析->过滤->返回列表 的工作
        filtered_sents_list, filtered_scores_list, _ = filter_judge.judge_sentence(raw_sents_str, raw_scores_str)
    except Exception as e:
        logger.error(f"Filter failed: {e}")
        filtered_sents_list, filtered_scores_list = [], []
    # 3. 将过滤后的列表再次格式化为字符串，以便存入 CSV
    filtered_sents_str = format_list_custom(filtered_sents_list)
    filtered_scores_str = format_list_custom(filtered_scores_list)

    # [新增] 在 format_list_custom 下面添加
    def format_context_list(items):
        """专门用于格式化上下文列表，使用分隔符避免混淆"""
        return "||||||||||||\n\n\n\n\n".join([str(item).replace("[", "(").replace("]", ")") for item in items])
    # [新增] 格式化 Contexts
    raw_contexts_str = format_context_list(collected_think_contexts)

    return (full_context, 
            raw_scores_str, raw_sents_str, raw_contexts_str,
            filtered_scores_str, filtered_sents_str)
    # return full_context, format_list_custom(collected_think_scores), format_list_custom(collected_think_sentences)


if __name__ == "__main__":
    # 1. 解析参数
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Model: {args.model_path}...")
    
    # 将 args.output_file 修改为model +dataset 命名以区分不同实验结果
    args.output_file = f"ours_{os.path.basename(args.model_path)}_{args.dataset}_results.csv"
    print(f"Results will be saved to: {args.output_file}")

    # 2. 初始化模型 (使用参数路径)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="auto", 
        local_files_only=True, attn_implementation="eager"
    )
    rind_calculator = RINDCalculator(model, tokenizer)

    print("Initializing Claim Filter...")
    filter_judge = BasicGeneratorRIND(
        use_api=(args.filter_use_api == "True"),
        api_base_url=args.api_base_url,
        api_key=args.api_key,
        api_model=args.api_model
    )
    
    # 定义主生成逻辑
    curr_eos = [151645, 151643] # for Qwen2.5 series models
    curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

    # 初始化停止条件
    target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
    stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])

    # 3. 加载数据集
    questions = []
    if args.dataset == "truthfulqa":
        print("Loading TruthfulQA...")
        ds = load_dataset("truthfulqa/truthful_qa", "generation")
        data = ds['validation']
        for item in data:
            questions.append(item['question'])
            
    elif args.dataset == "halueval":
        print(f"Loading HaluEval from {args.data_path}...")
        with open(args.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    questions.append(item['question'])
    
    print(f"Total questions: {len(questions)}")

    # 4. 准备 CSV 文件头 (如果文件不存在)
    if not os.path.exists(args.output_file):
        df_header = pd.DataFrame(columns=[
            "Full_Process", 
            "Think_RIND_Scores", 
            "Think_Sentences",
            "Think_Contexts",
            "Filtered_Think_Scores",   
            "Filtered_Think_Sentences" 
        ])
        df_header.to_csv(args.output_file, index=False)

    # 5. 处理循环 (带进度条)
    results_buffer = []
    
    for i, question in enumerate(tqdm(questions)):
        try:
            # 调用处理函数
            # full_ctx, scores_str, sents_str = process_single_question(
            #     question, model, tokenizer, rind_calculator, 
            #     stopping_criteria, curr_eos, curr_search_template, device
            # )
            # 加入缓存
            # results_buffer.append({
            #     "Full_Process": full_ctx,
            #     "Think_RIND_Scores": scores_str,
            #     "Think_Sentences": sents_str
            # })
            full_ctx, raw_scores, raw_sents, raw_ctxs, filt_scores, filt_sents = process_single_question(
                question, model, tokenizer, rind_calculator, 
                stopping_criteria, curr_eos, curr_search_template, device,
                filter_judge 
            )
            results_buffer.append({
                "Full_Process": full_ctx,
                "Think_RIND_Scores": raw_scores,
                "Think_Sentences": raw_sents,
                "Think_Contexts": raw_ctxs,
                "Filtered_Think_Scores": filt_scores,     
                "Filtered_Think_Sentences": filt_sents    
            })
            # 定期保存 (追加模式)
            if (i + 1) % args.save_freq == 0:
                df = pd.DataFrame(results_buffer)
                df.to_csv(args.output_file, mode='a', header=False, index=False)
                results_buffer = [] # 清空缓存
                
        except Exception as e:
            print(f"Error at index {i}: {e}")
            continue

    # 保存剩余数据
    if results_buffer:
        df = pd.DataFrame(results_buffer)
        df.to_csv(args.output_file, mode='a', header=False, index=False)

    print("Done.")