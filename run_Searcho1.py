import os
import argparse
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import copy
import requests
import logging
import numpy as np
import spacy
from scipy.special import softmax
from tqdm import tqdm
from datasets import load_dataset

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


# ==========================================
# 1. 之前代码中的辅助类 (请在此处粘贴)
# ==========================================


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
# 2. Search-o1 原生配置与 Prompt
# ==========================================

BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=str, default="6,7")
    parser.add_argument("--model_path", type=str, default="/hub/huggingface/models/Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", type=str, required=True, choices=["truthfulqa", "halueval"])
    parser.add_argument("--data_path", type=str, default="data/qa_data.json")
    parser.add_argument("--output_file", type=str, default="searcho1_results.csv")
    
    # RIND Filter 参数
    parser.add_argument("--filter_use_api", type=str, default="True")
    parser.add_argument("--api_base_url", type=str, default="http://10.98.36.100:8018/v1")
    parser.add_argument("--api_key", type=str, default="qiaoyt")
    parser.add_argument("--api_model", type=str, default="qwen2.5-72b-instruct")
    
    # Search O1 Params
    parser.add_argument('--MAX_SEARCH_LIMIT', type=int, default=5)
    parser.add_argument("--save_freq", type=int, default=1)
    
    return parser.parse_args()

def get_singleqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Who got the first Nobel Prize in Physics?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who was awarded the first Nobel Prize in Physics.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>first Nobel Prize in Physics winner<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )

def get_webpage_to_reasonchain_instruction(prev_reasoning, search_query, document):
    return f"""**Task Instruction:**

You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **Extract Relevant Information:**
- Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
- Ensure that the extracted information is accurate and relevant.

3. **Output Format:**
- **If the web pages provide helpful information for current search query:** Present the information beginning with `**Final Information**` as shown below.
**Final Information**

[Helpful information]

- **If the web pages do not provide any helpful information for current search query:** Output the following text.

**Final Information**

No helpful information found.

**Inputs:**
- **Previous Reasoning Steps:** {prev_reasoning}

- **Current Search Query:** {search_query}

- **Searched Web Pages:** {document}

Now you should analyze each web page and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
"""

def get_task_instruction_openqa(question):
    user_prompt = (
        'Please answer the following question. You should think step by step to solve it.\n\n'
        'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
        f'Question:\n{question}\n\n'
    )
    return user_prompt

# Search Function (Original)
def search(query: str):
    try:
        payload = {
            "queries": [query],
            "topk": 3,
            "return_scores": True
        }
        # 假设服务端口一致
        response = requests.post("http://10.98.36.100:8003/retrieve", json=payload, timeout=10)
        results = response.json()['result']
        
        def _passages2string(retrieval_result):
            format_reference = ''
            for idx, doc_item in enumerate(retrieval_result):
                content = doc_item['document']['contents']
                parts = content.split("\n")
                if len(parts) > 1:
                    title = parts[0]
                    text = "\n".join(parts[1:])
                else:
                    title = "Unknown"
                    text = content
                format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
            return format_reference

        if results and len(results) > 0:
            return _passages2string(results[0])
        return ""
    except Exception as e:
        logger.error(f"Search error: {e}")
        return ""

def extract_between(text: str, start_tag: str, end_tag: str):
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None

def extract_answer(output, mode='infogen'):
    extracted_text = ''
    if mode == 'infogen':
        pattern_info = "**Final Information**"
        if pattern_info in output:
            extracted_text = output.split(pattern_info)[-1].replace("\n"," ").strip("```").strip()
        else:
            extracted_text = "No helpful information found."
    return extracted_text

def format_list_custom(items):
    return "".join([f"[{str(item)}]" for item in items])

# ==========================================
# 3. 核心处理逻辑 (Search-o1 + RIND)
# ==========================================
def process_single_question_search_o1(question, model, tokenizer, rind_calculator, device, filter_judge, max_search_limit=5):
    
    # 1. 初始化 Prompt (Search-o1 风格)
    instruction = get_singleqa_search_o1_instruction(MAX_SEARCH_LIMIT=max_search_limit)
    user_prompt = get_task_instruction_openqa(question)
    
    # 构建初始 chat messages
    messages = [{"role": "user", "content": instruction + user_prompt}]
    
    # 2. 结果容器
    collected_sentences = []
    collected_scores = []
    collected_contexts = []
    
    # 3. 状态管理
    full_process_log = "" 
    current_search_context = "Internal Knowledge (No Search)" 
    
    finished = False
    search_count = 0
    executed_search_queries = set()
    current_round = 0
    MAX_TURN = 5
    
    # 当前对话历史 (用于 append)
    # Search-o1 原逻辑是不断 append text 到 prompt string。
    # 这里我们用 transformers，为了保持 KV Cache 效率或简单性，我们维护一个 messages 列表或者 text 
    # 为了 RIND 方便，每次都重新输入 full prompt 是计算最准确的 (虽然慢)。
    # Search-o1 这里采用的是 apply_chat_template 得到 string 后不断 += text。
    
    current_prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_process_log = current_prompt_str # 初始化日志
    
    while not finished and current_round < MAX_TURN:
        current_round += 1
        # print(f"--- Round {current_round} ---")
        
        # 准备输入
        model_inputs = tokenizer(current_prompt_str, return_tensors="pt").to(device)
        input_ids = model_inputs["input_ids"]
        
        # === A. 生成 (Main Reasoning) ===
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,      
            output_attentions=True,
            stop_strings=[END_SEARCH_QUERY, tokenizer.eos_token], 
            tokenizer=tokenizer 
        )
        
        # 提取新生成的 tokens
        gen_seq = outputs.sequences[0][input_ids.shape[1]:]
        gen_text_chunk = tokenizer.decode(gen_seq, skip_special_tokens=False) # 保留特殊token以便解析 tag
        
        # === B. RIND 计算 (Main Reasoning) ===
        if len(gen_seq) > 0:
            class MiniOut:
                def __init__(self, scores): self.scores = tuple(scores)
            mini_out = MiniOut(outputs.scores)
            full_input_ids = outputs.sequences
            
            try:
                # 传入 full_input_ids 进行 RIND 计算
                rind_list = rind_calculator.compute_rind_for_generation(mini_out, gen_seq, full_input_ids, solver='max')
                
                # 分句逻辑 (复用之前的)
                chunk_text_proper = tokenizer.decode(gen_seq, skip_special_tokens=True)
                doc = rind_calculator.nlp(chunk_text_proper)
                chunk_scores_list = [(w, s) for w, s, _, _, _, _ in rind_list]
                score_cursor = 0
                
                for span in doc.sents:
                    sent_text = span.text.strip()
                    if not sent_text: continue
                    
                    sent_pure_len = len(sent_text.replace(" ", "").replace("\n", ""))
                    current_sent_scores = []
                    acc_len = 0
                    temp_cursor = score_cursor
                    while temp_cursor < len(chunk_scores_list):
                        w, s = chunk_scores_list[temp_cursor]
                        w_clean = w.replace(rind_calculator.space_token, "").replace(" ", "").replace("\n", "")
                        current_sent_scores.append(s)
                        acc_len += len(w_clean)
                        temp_cursor += 1
                        if acc_len >= sent_pure_len: break
                    
                    score_cursor = temp_cursor
                    max_s = max(current_sent_scores) if current_sent_scores else 0.0
                    
                    collected_sentences.append(sent_text)
                    collected_scores.append(round(max_s, 4))
                    collected_contexts.append(current_search_context)
                    
            except Exception as e:
                logger.error(f"RIND Error: {e}")
                collected_sentences.append(gen_text_chunk.replace("\n", " "))
                collected_scores.append(0.0)
                collected_contexts.append(current_search_context)
        
        # 更新状态
        current_prompt_str += gen_text_chunk
        full_process_log += gen_text_chunk
        
        # === C. 检查 Search 意图 ===
        search_query = extract_between(gen_text_chunk, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
        
        if not search_query:
            # 没生成搜索请求，说明生成结束（或只生成了答案）
            finished = True
            break
            
        # === D. 执行搜索 (如果触发) ===
        if search_count >= max_search_limit:
            limit_msg = f"\n{BEGIN_SEARCH_RESULT}\nMax search limit exceeded.\n{END_SEARCH_RESULT}\n\n"
            current_prompt_str += limit_msg
            full_process_log += limit_msg
            continue
            
        # 调用搜索 API
        try:
            raw_docs = search(search_query)
        except Exception:
            raw_docs = ""
            
        search_count += 1
        executed_search_queries.add(search_query)
        
        # === E. Webpage -> ReasonChain (分析生成) ===
        # Search-o1 的特色：让 LLM 阅读网页并总结出 information
        # 我们需要构造一个新的 prompt，单独跑一次生成
        
        # 1. 构造截断的 reasoning history
        # 简单处理：把当前的 current_prompt_str 里 Assistant 的部分提取出来作为 reasoning steps
        # 这里为了简化，直接把 current_prompt_str 里的文本作为 context (或者按原代码逻辑 split)
        # 原代码：seq["output"].replace("\n\n", "\n").split("\n") -> truncated_reasoning
        # 我们这里 current_prompt_str 包含了 instruction，简单起见，我们传全部 prompt 作为 prev_reasoning 上下文
        
        analyze_instruction = get_webpage_to_reasonchain_instruction(
            prev_reasoning="[See context above]", # 简化，模型有上下文能力
            search_query=search_query,
            document=raw_docs
        )
        
        # 构造分析步骤的输入
        # 注意：Search-o1 原代码是 batch 生成，这里单条生成
        analyze_messages = [
            {"role": "user", "content": analyze_instruction}
        ]
        analyze_prompt_str = tokenizer.apply_chat_template(analyze_messages, tokenize=False, add_generation_prompt=True)
        
        analyze_inputs = tokenizer(analyze_prompt_str, return_tensors="pt").to(device)
        
        # 生成分析 (同样计算 RIND，因为这也是模型生成的思考内容)
        analyze_outputs = model.generate(
            **analyze_inputs,
            max_new_tokens=1024, # 分析可能长一点
            do_sample=True,
            temperature=0.7,
            output_scores=True,
            output_attentions=True,
            return_dict_in_generate=True
        )
        
        analyze_gen_seq = analyze_outputs.sequences[0][analyze_inputs['input_ids'].shape[1]:]
        analyze_text_chunk = tokenizer.decode(analyze_gen_seq, skip_special_tokens=True)
        
        # RIND 计算 (Analysis Phase)
        if len(analyze_gen_seq) > 0:
            # 复用计算逻辑...
            try:
                mini_out_a = MiniOut(analyze_outputs.scores)
                full_input_ids_a = analyze_outputs.sequences
                rind_list_a = rind_calculator.compute_rind_for_generation(mini_out_a, analyze_gen_seq, full_input_ids_a, solver='max')
                
                # 分句收集
                doc_a = rind_calculator.nlp(analyze_text_chunk)
                chunk_scores_list_a = [(w, s) for w, s, _, _, _, _ in rind_list_a]
                score_cursor_a = 0
                for span in doc_a.sents:
                    sent_text = span.text.strip()
                    if not sent_text: continue
                    
                    # ... (同上分句对齐逻辑) ...
                    sent_pure_len = len(sent_text.replace(" ", "").replace("\n", ""))
                    current_sent_scores = []
                    acc_len = 0
                    temp_cursor = score_cursor_a
                    while temp_cursor < len(chunk_scores_list_a):
                        w, s = chunk_scores_list_a[temp_cursor]
                        w_clean = w.replace(rind_calculator.space_token, "").replace(" ", "").replace("\n", "")
                        current_sent_scores.append(s)
                        acc_len += len(w_clean)
                        temp_cursor += 1
                        if acc_len >= sent_pure_len: break
                    score_cursor_a = temp_cursor
                    max_s = max(current_sent_scores) if current_sent_scores else 0.0
                    
                    collected_sentences.append(f"[Analysis] {sent_text}") # 标记这是分析内容
                    collected_scores.append(round(max_s, 4))
                    # 这个阶段的上下文就是刚才搜到的文档
                    collected_contexts.append(f"[Analysis Context] {raw_docs[:200]}...")
            except Exception as e:
                logger.error(f"RIND Analysis Error: {e}")
                
        # 提取 Final Information
        extracted_info = extract_answer(analyze_text_chunk, mode='infogen')
        
        # === F. 将结果拼接到主 Prompt ===
        append_text = f"\n\n{BEGIN_SEARCH_RESULT}\n**Final Information**\n{extracted_info}\n{END_SEARCH_RESULT}\n\n"
        
        current_prompt_str += append_text
        full_process_log += append_text
        
        # 更新上下文变量，供下一轮 Main Reasoning 使用
        current_search_context = f"[Query: {search_query}] {extracted_info}"

    # === Filter 逻辑 ===
    raw_sents_str = format_list_custom(collected_sentences)
    raw_scores_str = format_list_custom(collected_scores)
    
    try:
        filtered_sents_list, filtered_scores_list, _ = filter_judge.judge_sentence(raw_sents_str, raw_scores_str)
    except Exception as e:
        logger.error(f"Filter failed: {e}")
        filtered_sents_list, filtered_scores_list = [], []
        
    filtered_sents_str = format_list_custom(filtered_sents_list)
    filtered_scores_str = format_list_custom(filtered_scores_list)
    
    def format_context_list(items):
        return "||||||||||||\n\n\n\n\n".join([str(item).replace("[", "(").replace("]", ")") for item in items])
    raw_contexts_str = format_context_list(collected_contexts)
    
    return (full_process_log, 
            raw_scores_str, raw_sents_str, raw_contexts_str,
            filtered_scores_str, filtered_sents_str)

# ==========================================
# 4. Main
# ==========================================
def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading Model: {args.model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    # 必须使用 Transformers 才能支持 RIND 计算 (vllm 不支持 output_attentions)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        output_attentions=True 
    )
    
    # 实例化你粘贴进来的类
    rind_calculator = RINDCalculator(model, tokenizer)
    filter_judge = BasicGeneratorRIND(
        use_api=(args.filter_use_api == "True"),
        api_base_url=args.api_base_url,
        api_key=args.api_key,
        api_model=args.api_model
    )

    # 数据加载 (复用 SearchR1/DRAGIN 逻辑)
    dataset = []
    if "truthfulqa" in args.dataset.lower():
        print("Loading TruthfulQA...")
        ds = load_dataset("truthfulqa/truthful_qa", "generation")
        for item in ds['validation']:
            dataset.append(item['question'])
    elif "halueval" in args.dataset.lower():
        print(f"Loading HaluEval from {args.data_path}...")
        with open(args.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    dataset.append(data['question'])
    
    print(f"Total questions: {len(dataset)}")
    model_base_name = args.model_path.split("/")[-1]
    args.output_file = model_base_name +  "_" + args.dataset + "_" + args.output_file

    # CSV Header
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
        
    results_buffer = []
    
    for i, question in enumerate(tqdm(dataset)):
        try:
            full_ctx, raw_scores, raw_sents, raw_ctxs, filt_scores, filt_sents = process_single_question_search_o1(
                question, model, tokenizer, rind_calculator, 
                device, filter_judge, max_search_limit=args.MAX_SEARCH_LIMIT
            )
            
            results_buffer.append({
                "Full_Process": full_ctx,
                "Think_RIND_Scores": raw_scores,
                "Think_Sentences": raw_sents,
                "Think_Contexts": raw_ctxs,
                "Filtered_Think_Scores": filt_scores,     
                "Filtered_Think_Sentences": filt_sents    
            })
            
            if (i + 1) % args.save_freq == 0:
                df = pd.DataFrame(results_buffer)
                df.to_csv(args.output_file, mode='a', header=False, index=False)
                results_buffer = []
                
        except Exception as e:
            logger.error(f"Error processing index {i}: {e}")
            continue

    if results_buffer:
        df = pd.DataFrame(results_buffer)
        df.to_csv(args.output_file, mode='a', header=False, index=False)
    
    print("All Done.")

if __name__ == "__main__":
    main()