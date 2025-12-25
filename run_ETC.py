import os
import json
import logging
import argparse
import string
import spacy
import torch
import requests
from typing import List, Tuple, Dict, Union
from dataclasses import dataclass
from copy import copy
from tqdm import tqdm
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, StoppingCriteria, StoppingCriteriaList
from transformers.generation.utils import GenerateDecoderOnlyOutput
from datasets import load_dataset

# 设置环境变量，防止 Tokenizer 并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局加载 Spacy 模型（ETC 依赖此分词）
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("正在下载 spacy 模型 en_core_web_sm...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

DEBUG = False 

# ==========================================
# Part 1: 数据结构与工具函数
# ==========================================

@dataclass
class Block:
    text: str = None
    tokens: List[str] = None
    range_: List[Tuple[int, int]] = None
    @property
    def len_tokens(self):
        return len(self.tokens)
    @property
    def len_words(self):
        return len(self.range_)

def merge_blocks(blocks: List[Block]) -> Block:
    text = "".join([block.text for block in blocks])
    tokens = sum([block.tokens for block in blocks], [])
    range_ = []
    st = 0
    for block in blocks:
        if block.range_:
            for l, r in block.range_:
                range_.append((st+l, st+r))
            st = range_[-1][1]
    return Block(text=text, tokens=tokens, range_=range_)

class Counter:
    def __init__(self):
        self.retrieve = 0
        self.generate = 0
        self.hallucinated = 0
        self.token = 0
        self.sentence = 0

    def add_generate(self, text, tokenizer):
        self.generate += 1
        ids = tokenizer(text, return_tensors="pt")['input_ids'][0].tolist()
        self.token += len(ids)
        sentences = [sent.text for sent in nlp(text).sents]
        self.sentence += len(sentences)

    def calc(self, other_counter):
        return {
            "retrieve_count": self.retrieve - other_counter.retrieve,
            "generate_count": self.generate - other_counter.generate,
            "hallucinated_count": self.hallucinated - other_counter.hallucinated,
            "token_count": self.token - other_counter.token,
            "sentence_count": self.sentence - other_counter.sentence
        }

@dataclass
class GeneratorOutput:
    ended: bool
    empty: bool
    blocks: List[Block] = None
    merged_blocks: Block = None
    atten: Tensor = None
    max_atten: Tensor = None
    entropies: Tensor = None
    entropies_s1: Tensor = None
    entropies_s2: Tensor = None
    smooth_s2: Tensor = None
    mt_s2: Tensor = None
    fun_word: Tensor = None
    @property
    def new_text(self):
        return self.blocks[-1].text
    @property
    def len_new_words(self):
        return self.blocks[-1].len_words

@dataclass
class CheckerOutput:
    hallucination: bool
    curr_st: int = None
    curr_en: int = None
    curr_thres: List[bool] = None

def join_if_nonempty(*li, sep=" "):
    return sep.join([s for s in li if len(s) > 0])

def match(word: str, real_words):
    for real_word in real_words:
        if real_word in word:
            return True
    return False

def get_top_sentence(text):
    prev = ""
    for sent in nlp(text).sents:
        prev += sent.text
        sent = sent.text.strip()
        if len(sent) > 0:
            return prev
    return ""

# ==========================================
# Part 2: Generator 类 
# ==========================================

# 自定义停止条件，替代 buggy 的 stop_strings
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 检查生成的最后一个 token 是否在停止列表中
        if input_ids[0][-1] in self.stop_token_ids:
            return True
        return False

class Generator:
    def __init__(self, model_name_or_path: str):
        logger.info(f"Loading model from {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            device_map="auto", 
            torch_dtype=torch.float16
        )
        self.model.eval()
        logger.info(f"device = {self.model.device}")
        
        # 识别换行符对应的 token id
        # Llama 3 中 '\n' 是一般是 198 ('Ċ')
        self.newline_token_id = self.tokenizer.encode("\n", add_special_tokens=False)[-1]
        logger.info(f"Detected newline token id: {self.newline_token_id}")
        
        self.space_token = "Ġ" if "llama-3" in model_name_or_path.lower() else " " 
        if "llama-2" in model_name_or_path.lower():
             self.space_token = " " 

        self.tokens_cannot_merged = {
            self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode("0" + ch)[-1:])[0]
            for ch in string.whitespace + string.punctuation
        } | {self.space_token, self.tokenizer.bos_token, self.tokenizer.eos_token}

    def simply_generate(self, input_text: str, max_length: int) -> Tuple[bool, str]:
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)
        
        # 使用自定义停止条件
        stopping_criteria = StoppingCriteriaList([StopOnTokens([self.newline_token_id])])
        
        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            stopping_criteria=stopping_criteria, # 替代 stop_strings
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.pad_token_id
        )[0, input_length:]
        
        if output_ids.shape[0] == 0:
            return True, ""
        if output_ids[0] == self.tokenizer.bos_token_id:
            output_ids = output_ids[1:]
        
        text = self.tokenizer.decode(output_ids)
        
        # 检查是否以 EOS 结束
        if output_ids[-1] == self.tokenizer.eos_token_id:
            return True, self.tokenizer.decode(output_ids[:-1])
            
        return False, text

    def tokenize(self, text: str, is_start: bool = False):
        ids = self.tokenizer.encode(text)
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        if not is_start and len(tokens) > 0 and tokens[0] == self.tokenizer.bos_token:
            tokens = tokens[1:]
        return tokens

    def merge_tokens(self, tokens) -> List[Tuple[int, int]]:
        range_ = []
        for i, t in enumerate(tokens):
            if i == 0 or t.startswith(self.space_token) \
                or tokens[i] in self.tokens_cannot_merged \
                or tokens[i-1] in self.tokens_cannot_merged:
                range_.append([i, i+1])
            else:
                range_[-1][1] += 1
        return range_

    def build_block(self, text: str, is_start: bool = False) -> Block:
        tokens = self.tokenize(text, is_start=is_start)
        range_ = self.merge_tokens(tokens)
        return Block(text=text, tokens=tokens, range_=range_)

    def generate(self, input_texts: List[str], max_length: int) -> GeneratorOutput:
        blocks = []
        for text in input_texts:
            blocks.append(self.build_block(text, is_start=not blocks))

        input_tokens = sum([block.tokens for block in blocks], [])
        input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(input_tokens)], device=self.model.device)
        input_len_tokens = len(input_tokens)
        attention_mask = torch.ones_like(input_ids)

        # 使用自定义停止条件
        stopping_criteria = StoppingCriteriaList([StopOnTokens([self.newline_token_id])])

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            return_dict_in_generate=True,
            output_scores=True,
            stopping_criteria=stopping_criteria, # 替代 stop_strings
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.pad_token_id
        )
        outputs: GenerateDecoderOnlyOutput

        tokens = self.tokenizer.convert_ids_to_tokens(outputs.sequences[0, input_len_tokens:])
        
        if len(tokens) <= 1:
            return GeneratorOutput(ended=True, empty=True)

        ended = (tokens[-1] == self.tokenizer.eos_token)
        if ended:
            tokens = tokens[:-1]
        text = self.tokenizer.convert_tokens_to_string(tokens)
        range_ = self.merge_tokens(tokens)
        new_block = Block(text=text, tokens=tokens, range_=range_)

        blocks.append(new_block)
        merged_blocks = merge_blocks(blocks)

        atten = self.model(outputs.sequences, output_attentions=True).attentions[-1][0][:, -new_block.len_tokens:, :]
        atten = atten.mean(dim=0)
        atten = torch.stack([atten[:, l:r].sum(dim=-1) for l, r in merged_blocks.range_], dim=-1)
        atten = torch.stack([atten[l:r, :].mean(dim=-2) for l, r in range_], dim=-2)

        atten_to_new = atten[:, -new_block.len_words:]
        atten_to_new /= atten.sum(dim=-1,keepdim=True) + 1e-10
        max_atten, _ = atten_to_new.max(dim=1)

        probs = torch.stack(outputs.scores).softmax(dim=-1)
        entropies = (-probs * torch.log(probs + 1e-10)).sum(dim=-1)
        
        # 安全计算 Word 级熵
        word_entropies = []
        for l, r in range_:
            entropy_slice = entropies[l:r, 0]
            if entropy_slice.numel() > 0:
                word_entropies.append(entropy_slice.max())
            else:
                word_entropies.append(torch.tensor(0.0, device=entropies.device, dtype=entropies.dtype))
        
        entropies = torch.stack(word_entropies)

        func_words=[]   
        doc = nlp(new_block.text)
        real_words = set(token.text for token in doc if token.pos_ in ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
        wl = 0
        wr = new_block.len_words
        for i in range(wl, wr):
            tl, tr = new_block.range_[i]
            word = self.tokenizer.convert_tokens_to_string(new_block.tokens[tl:tr])
            if not match(word, real_words):
                func_words.append(i)
                
        entropies_s1 = [{'key': i, 'val': torch.tensor(0, dtype=torch.float64)} for i in range(len(range_))]
        entropies_s2 = [{'key': i, 'val': torch.tensor(0, dtype=torch.float64)} for i in range(len(range_))]
        smooth_s2 = [{'key': i, 'val': torch.tensor(0, dtype=torch.float64)} for i in range(len(range_))]
        mt_s2 = [{'key': i, 'val': torch.tensor(0, dtype=torch.float64)} for i in range(len(range_))]
        fun_word = [{'key': i, 'val': torch.tensor(0, dtype=torch.float64)} for i in range(len(range_))]

        for i, (l,r) in enumerate(range_[:]):
            if i not in func_words:
                fun_word[i]['val'] = torch.tensor(1, dtype=torch.float64)

        for i, (l, r) in enumerate(range_[1:]): 
            if i+1 not in func_words:
                j = i
                while j >= 0:
                    if j not in func_words:
                        s1 = (entropies[i+1].to(torch.float64) - entropies[j].to(torch.float64))
                        entropies_s1[i+1]['val'] = s1
                        break
                    if j == 0: break
                    else: j -= 1
        
        for i, (l, r) in enumerate(range_[2:]): 
            if i+2 not in func_words: 
                j = i + 1
                while j >= 1:
                    if entropies_s1[j]['val'].item() != 0: 
                        s2 = (entropies_s1[i+2]['val'].to(torch.float64) - entropies_s1[j]['val'].to(torch.float64)) 
                        entropies_s2[i+2]['val'] = s2
                        break
                    if j == 1: break
                    else: j -= 1

        count_fun = 0
        sum_s2 = 0
        Mt_1 = torch.tensor(0, dtype=torch.float64) 
        for i, (l, r) in enumerate(range_[2:]):
            if entropies_s2[i+2]['val'] != 0:
                count_fun +=1 
                sum_s2 += entropies_s2[i+2]['val'].item()
                s2_mean = sum_s2/count_fun 
                w = torch.abs((Mt_1 - s2_mean)) /(torch.abs((entropies_s2[i+2]['val']-s2_mean)) + torch.abs((Mt_1 - s2_mean)))
                α = 0.9 + 0.1 * w
                Mt = α * entropies_s2[i+2]['val'] + (1-α) * Mt_1
                mt_s2[i+2]['val'] = Mt
                Mt_1 = entropies_s2[i+2]['val']
            elif entropies_s1[i+2]['val'].item() != 0:
                pass 

        return GeneratorOutput(
            empty = False,
            ended=ended,
            blocks=blocks,
            merged_blocks=merged_blocks,
            atten=atten,
            max_atten=max_atten,
            entropies=entropies,
            entropies_s1 = entropies_s1,
            entropies_s2 = entropies_s2,
            smooth_s2 = smooth_s2,
            mt_s2 = mt_s2,
            fun_word = fun_word,
        )

# ==========================================
# Part 3: 自定义 Retriever
# ==========================================

class CustomRetriever:
    def __init__(self):
        self.api_url = "http://10.98.36.100:8003/retrieve"

    def __call__(self, query: str, topk: int = 3) -> List[str]:
        payload = {
            "queries": [query],
            "topk": topk,
            "return_scores": True
        }
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            results = response.json().get('result', [])
            
            if not results:
                return []
            
            docs = []
            for doc_item in results[0]:
                content = doc_item['document']['contents']
                docs.append(content)
                
            return docs
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []

# ==========================================
# Part 4: ETC 核心类
# ==========================================

class ETC:
    def __init__(self, args):
        for k, v in args.__dict__.items():
            setattr(self, k, v)
        self.generator = Generator(self.model_name_or_path)
        self.tokenizer = self.generator.tokenizer
        self.model = self.generator.model 
        self.retriever = CustomRetriever()
        self.counter = Counter()

    def hallucination_check(self, outputs: GeneratorOutput) -> CheckerOutput: 
        if DEBUG: print("Start detecting hallucinations")
        new_block = outputs.blocks[-1]
        sentences = [sent.text.strip() for sent in nlp(new_block.text).sents]   
        sentences = [sent for sent in sentences if len(sent) > 0]  
        
        wid = 0
        word_counts = [0] * len(sentences)

        for sid, sent in enumerate(sentences):  
            wl, wr = wid, wid 
            if wid == new_block.len_words:
                break
            while wr < new_block.len_words:
                 token_seg = self.tokenizer.convert_tokens_to_string(
                     new_block.tokens[new_block.range_[wl][0]:new_block.range_[wr][1]]
                 )
                 if sent in token_seg:
                     break
                 wr += 1
                 
            if wr < new_block.len_words:
                wr += 1 
            wid = wr    
            len_sent = wid
            if wl == wr:
                continue
            if sid == 0:    
                word_counts[sid] = wid
            else:
                for t in range(0,sid):
                    len_sent -= word_counts[t]
                word_counts[sid] = len_sent 
            
            max_atten_sent = outputs.max_atten[wl: wr]
            max_atten_sent = max_atten_sent * (wr - wl) / (max_atten_sent.sum() + 1e-10)
            
            value = max_atten_sent * torch.tensor([entry['val'] for entry in outputs.mt_s2[wl: wr]]).to(max_atten_sent.device) 
            thres_abs = self.thres_abs
            if thres_abs == True:
                thres = (torch.abs(value) > self.hallucination_threshold)
            else:
                thres = (value > self.hallucination_threshold)

            if True in thres:
                for i in range(wl, wr):
                    if thres[i-wl].item() == True:
                        count_k_2 = 0
                        j = i - 1
                        while(count_k_2 < 2):
                            if j < 0: break 
                            if outputs.fun_word[j]['val'].item() != 0:
                                count_k_2 += 1
                            if count_k_2 == 2:
                                break
                            else:
                                j -= 1
                        return CheckerOutput(hallucination=True, curr_st=i, curr_en=wr, curr_thres=thres[i-wl:wr]) 
                    
        return CheckerOutput(hallucination=False)

    def generate_retrieve_qry(self, outputs: GeneratorOutput, check_info: CheckerOutput):
        try:
            curr_st, curr_en = check_info.curr_st, check_info.curr_en
            text_atten = outputs.atten[curr_st:curr_en, :curr_st]
            
            if check_info.curr_thres.shape[0] == text_atten.shape[0]:
                text_atten = text_atten[check_info.curr_thres, :].sum(dim=0)
            else:
                text_atten = text_atten.sum(dim=0) 

            doc = nlp(outputs.merged_blocks.text)
            real_words = set(token.text for token in doc if token.pos_ in 
                    ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])

            real_pairs = []
            for i in range(text_atten.shape[0]):
                a = text_atten[i]
                if i < len(outputs.merged_blocks.range_):
                    tl, tr = outputs.merged_blocks.range_[i]
                    word = self.tokenizer.convert_tokens_to_string(outputs.merged_blocks.tokens[tl:tr])
                    if match(word, real_words):
                        real_pairs.append((a, word, i)) 
            
            top_k = 35 
            if "retrieve_keep_top_k" in self.__dict__:
                top_k = min(self.retrieve_keep_top_k, len(real_pairs))
            
            real_pairs.sort(key=lambda x: -x[0])    
            real_pairs = real_pairs[:top_k]    
            real_pairs.sort(key=lambda x: x[2])    

            return " ".join([x[1] for x in real_pairs]) 
        except Exception as e:
            logger.warning(f"Query formulation failed: {e}, falling back to last sentence.")
            return outputs.new_text 

    def inference(self, question, demo=""):
        ##### 额外添加指令
        # format_instruction = " Please enclose your final answer in <answer> and </answer> tags."
        format_instruction = " First provide your reasoning. Then provide the final direct answer enclosed inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>."

        text = ""
        # 优化 Prompt 构建逻辑
        input_prompts = []
        if demo and len(demo.strip()) > 0: 
            input_prompts.append(demo)
        
        # input_prompts.extend(["\nQuestion:", question, "\nAnswer:", text])
        input_prompts.extend([f"\nQuestion: {question}{format_instruction if not demo else ''}", "\nAnswer:", text])
        
        if DEBUG: print("Begin reasoning")
        
        while True:
            old_len = len(text)
            # 构建当前的 prompts，text 是累积生成的回答
            # if demo and len(demo.strip()) > 0:
            #     current_prompts = [demo, "\nQuestion:", question, "\nAnswer:", text]
            # else:
            #     current_prompts = ["Question:", question, "\nAnswer:", text]
            if demo and len(demo.strip()) > 0:
                # Few-shot 模式：依赖 demo 里的格式
                current_prompts = [demo, f"\nQuestion: {question}", "\nAnswer:", text]
            else:
                # Zero-shot 模式：显式指令
                current_prompts = [f"Question: {question}{format_instruction}", "\nAnswer:", text]

            outputs = self.generator.generate(
                input_texts=current_prompts,
                max_length=self.generate_max_length,
            )

            if outputs.empty:
                break

            check_info = self.hallucination_check(outputs)
            
            if not check_info.hallucination:
                if DEBUG: print("No hallucinations")
                text = join_if_nonempty(text, outputs.new_text.strip())
                
                if "</answer>" in text:
                    break

                if outputs.ended or outputs.merged_blocks.len_tokens > self.generate_max_length:
                    break
            else:
                if DEBUG: print("Hallucination detected. Preparing to retrieve information.")
                retrieve_qry = self.generate_retrieve_qry(outputs, check_info)  
                if DEBUG: print(f"retrieve_qry: {retrieve_qry}")
                
                docs = self.retriever(retrieve_qry, topk=self.retrieve_topk)    
                self.counter.retrieve += 1
                
                # prompt = demo + "\n" if demo else ""
                # prompt += "Context:\n"
                # for i, doc in enumerate(docs):
                #     prompt += f"[{i+1}] {doc}\n"
                # prompt += "Answer in the same format as before.\n"
                # prompt += f"Question: {question}\nAnswer:"
                # print(prompt)
                prompt = demo + "\n" if demo else ""
                prompt += "Context:\n"
                for i, doc in enumerate(docs):
                    prompt += f"[{i+1}] {doc}\n"
                # 根据是否有 demo 决定提示词
                if demo:
                    prompt += "Answer in the same format as before.\n"
                else:
                    # prompt += "Based on the context, please answer the question. Enclose your final answer in <answer> and </answer> tags.\n"
                    prompt += "Based on the context, please answer the question. First provide your reasoning. Then provide the final direct answer enclosed inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>."
                prompt += f"Question: {question}\nAnswer:"

                current_block_text_before_hallucination = ""
                # 安全获取幻觉前的文本
                if outputs.blocks and check_info.curr_st < len(outputs.blocks[-1].range_):
                     range_start = outputs.blocks[-1].range_[check_info.curr_st][0]
                     current_block_text_before_hallucination = self.tokenizer.convert_tokens_to_string(
                        outputs.blocks[-1].tokens[:range_start]
                     )
                
                prompt += text + current_block_text_before_hallucination
                
                ended, new_texts = self.generator.simply_generate(
                    prompt, 
                    max_length=self.generate_max_length,
                )
                
                if self.use_counter:
                    self.counter.add_generate(new_texts, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                
                new_text = get_top_sentence(new_texts)
                text = join_if_nonempty(text, current_block_text_before_hallucination, new_text.strip())
                
                if "</answer>" in text:
                    break

                if DEBUG: print("Regenerate new text:", new_text, sep="\n")
                
                if len(self.tokenizer.encode(text)) > self.generate_max_length:
                    break
            
            if old_len >= len(text):
                break
                
        return text

# ==========================================
# Part 5: Main Logic
# ==========================================

def construct_few_shot_demo(data, shot_num):
    demo_str = ""
    for i in range(min(shot_num, len(data))):
        item = data[i]
        q = item['question']
        a = item['best_answer']
        demo_str += f"Question: {q}\nAnswer: {a}\n\n"
    return demo_str.strip()

def load_and_split_data(args):
    dataset_items = []
    
    if args.dataset == "truthfulqa":
        print("Loading TruthfulQA...")
        try:
            ds = load_dataset("truthfulqa/truthful_qa", "generation")
            data = ds['validation']
            for item in data:
                dataset_items.append({
                    "qid": str(len(dataset_items)), 
                    "question": item['question'],
                    "best_answer": item['best_answer']
                })
        except Exception as e:
            logger.error(f"Error loading TruthfulQA: {e}")
            
    elif args.dataset == "halueval":
        print(f"Loading HaluEval from {args.data_path}...")
        try:
            with open(args.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        ans = item.get('right_answer', item.get('answer', ''))
                        dataset_items.append({
                            "qid": str(len(dataset_items)), 
                            "question": item['question'],
                            "best_answer": ans
                        })
        except Exception as e:
            logger.error(f"Error loading HaluEval: {e}")
    
    demo_text = ""
    test_data = dataset_items
    
    if args.fewshot > 0:
        print(f"Constructing {args.fewshot}-shot demo...")
        demo_text = construct_few_shot_demo(dataset_items, args.fewshot)

    print(f"Total questions: {len(test_data)}")
    return test_data, demo_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--dataset", type=str, required=True, choices=["truthfulqa", "halueval"])
    parser.add_argument("--data_path", type=str, default="./data/halueval_data.jsonl", help="Path for JSONL data")
    parser.add_argument("--output_dir", type=str, default="./etc_results")
    
    # ETC 算法参数
    parser.add_argument("--generate_max_length", type=int, default=512)
    parser.add_argument("--retrieve_keep_top_k", type=int, default=35)
    parser.add_argument("--retrieve_topk", type=int, default=3)
    parser.add_argument("--hallucination_threshold", type=float, default=1.3)
    parser.add_argument("--check_real_words", type=bool, default=True)
    parser.add_argument("--use_counter", type=bool, default=True)
    parser.add_argument("--thres_abs", type=bool, default=False)
    parser.add_argument("--fewshot", type=int, default=0, help="Number of few-shot examples")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    model = ETC(args)
    data, demo_text = load_and_split_data(args)
    
    output_filename = f"{args.dataset}_etc_results.jsonl"
    output_path = os.path.join(args.output_dir, output_filename)
    
    print(f"Running inference... Output will be saved to {output_path}")
    if args.fewshot > 0:
        print(f"Few-shot Demo:\n{demo_text[:200]}...\n(Truncated)")

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in tqdm(data):
            qid = entry['qid']
            question = entry['question']
            
            try:
                last_counter = copy(model.counter)
                pred = model.inference(question, demo=demo_text)
                
                ret = {
                    "qid": qid,
                    "question": question,
                    "answer": pred.strip()
                }
                
                if args.use_counter:
                    ret.update(model.counter.calc(last_counter))
                
                f.write(json.dumps(ret) + "\n")
                f.flush() 
            except Exception as e:
                logger.error(f"Error processing qid {qid}: {e}")
                import traceback
                traceback.print_exc() 
                continue

if __name__ == "__main__":
    with torch.no_grad():
        main()