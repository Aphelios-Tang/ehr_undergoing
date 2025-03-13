import os
import re
import torch
import transformers
from transformers import AutoModelForCausalLM, LlamaTokenizer
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import tqdm
from rouge import Rouge
import random


from icformer import (
    ICFormerConfig, 
    ICFormerModel, 
)
from modules import Trainer, ICFormer
from data_utils import PileDataset, MIMICDataset
# from my_mimic_dataset_for_reconstruction import MIMIC_Dataset
from utils import parse_args, seed_everything

nltk_data_path = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/longcontext/model/nltk"
nltk.data.path.append(nltk_data_path)

SEED = 42
transformers.set_seed(SEED)
IGNORE_INDEX = -100

def extract_numbers(text):
    """从文本中提取所有数字"""
    # 匹配整数和小数
    pattern = r'\b\d+(?:\.\d+)?\b'
    return re.findall(pattern, text)

def calculate_number_accuracy(reference, hypothesis):
    """计算数字重建的准确率"""
    # 提取原文和预测文本中的数字
    ref_numbers = extract_numbers(reference)
    hyp_numbers = extract_numbers(hypothesis)
    
    if len(ref_numbers) == 0:
        return 1.0  # 如果原文中没有数字，则认为准确率为100%
    
    # 计算匹配的数字数量
    matched = sum(1 for n in hyp_numbers if n in ref_numbers)
    
    # 计算准确率
    accuracy = matched / len(ref_numbers)
    return accuracy

def calculate_bleu(reference, hypothesis):
    """计算不同n-gram的BLEU分数"""
    # 分词
    reference_tokens = word_tokenize(reference.lower())
    hypothesis_tokens = word_tokenize(hypothesis.lower())
    
    # 使用平滑函数避免零精度问题
    smoothie = SmoothingFunction().method1
    
    # 计算BLEU分数
    reference_list = [reference_tokens]
    
    bleu1 = sentence_bleu(reference_list, hypothesis_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu2 = sentence_bleu(reference_list, hypothesis_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu3 = sentence_bleu(reference_list, hypothesis_tokens, weights=(0.33, 0.33, 0.34, 0), smoothing_function=smoothie)
    bleu4 = sentence_bleu(reference_list, hypothesis_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    
    return bleu1, bleu2, bleu3, bleu4

def calculate_rouge(reference, hypothesis):
    """计算ROUGE分数"""
    try:
        rouge = Rouge()
        scores = rouge.get_scores(hypothesis, reference)
        
        # 提取F1分数
        rouge1 = scores[0]['rouge-1']['f']
        rouge2 = scores[0]['rouge-2']['f']
        rougeL = scores[0]['rouge-l']['f']
        
        return rouge1, rouge2, rougeL
    except Exception as e:
        print(f"计算ROUGE出错: {e}")
        return 0, 0, 0
    
def process_sample(model, tokenizer, context, max_new_tokens=512):
    """处理单个样本并返回重建文本"""

    context = context[:1024]

    context_ids = model.tokenizer(context)['input_ids']

    if len(context_ids) > model.max_seq_len: # random split
        last_start = len(context_ids) - model.max_seq_len
        start = random.randint(0, last_start)
        context_ids = context_ids[start:start+model.max_seq_len]

    context_embeds = model.convert_ids_to_embeds(context_ids)
    
    soft_prompt = model.get_soft_prompt(inputs_embeds=context_embeds, use_chunk=model.use_chunk)
    # pre_embeds = model.convert_ids_to_embeds(model.tokenizer(model.tokenizer.bos_token)['input_ids'])

    if model.encode:
        with torch.no_grad():
            context_embeds = model.language_model.model(inputs_embeds=context_embeds)[0]

    # inputs_embeds = torch.cat([pre_embeds, soft_prompt, prompt_embeds], dim=1)
    # inputs_embeds = torch.cat([pre_embeds, soft_prompt, model.AE], dim=1)
    inputs_embeds = torch.cat([soft_prompt, model.AE], dim=1)
    
    outputs = model.language_model.generate(
        inputs_embeds=inputs_embeds, 
        max_new_tokens=2048,
        do_sample=False,
        top_k=50
    )
    
    prediction = model.tokenizer.decode(outputs[0], False)
    return prediction

def evaluate_dataset(model, dataset, num_samples=None, max_new_tokens=512):
    """评估整个数据集的BLEU和ROUGE分数"""
    if num_samples is None:
        num_samples = len(dataset)
    else:
        num_samples = min(num_samples, len(dataset))
    
    # 用于存储所有样本的分数
    all_bleu1, all_bleu2, all_bleu3, all_bleu4 = [], [], [], []
    all_rouge1, all_rouge2, all_rougeL = [], [], []
    all_number_accuracies = [] 
    
    # 保存所有的原文本和预测文本，用于后续分析
    all_sources = []
    all_predictions = []
    
    print(f"开始评估 {num_samples} 个样本...")
    for i in tqdm.tqdm(range(num_samples)):
        context = dataset[i]
        all_sources.append(context)
        
        # 获取预测文本
        prediction = process_sample(model, model.tokenizer, context, max_new_tokens)
        all_predictions.append(prediction)
        
        # 计算评分
        try:
            bleu1, bleu2, bleu3, bleu4 = calculate_bleu(context, prediction)
            all_bleu1.append(bleu1)
            all_bleu2.append(bleu2)
            all_bleu3.append(bleu3)
            all_bleu4.append(bleu4)
            
            rouge1, rouge2, rougeL = calculate_rouge(context, prediction)
            all_rouge1.append(rouge1)
            all_rouge2.append(rouge2)
            all_rougeL.append(rougeL)
            # 计算数字准确率
            number_accuracy = calculate_number_accuracy(context, prediction)
            all_number_accuracies.append(number_accuracy)
            
            print(f"样本 {i}:")
            print(f"  BLEU-1: {bleu1:.4f}, BLEU-2: {bleu2:.4f}, BLEU-3: {bleu3:.4f}, BLEU-4: {bleu4:.4f}")
            print(f"  ROUGE-1: {rouge1:.4f}, ROUGE-2: {rouge2:.4f}, ROUGE-L: {rougeL:.4f}")
            print(f"  数字准确率: {number_accuracy:.4f}")
        except Exception as e:
            print(f"样本 {i} 评分出错: {e}")
    
    # 计算平均分数
    avg_bleu1 = np.mean(all_bleu1)
    avg_bleu2 = np.mean(all_bleu2)
    avg_bleu3 = np.mean(all_bleu3)
    avg_bleu4 = np.mean(all_bleu4)
    
    avg_rouge1 = np.mean(all_rouge1)
    avg_rouge2 = np.mean(all_rouge2)
    avg_rougeL = np.mean(all_rougeL)

    avg_number_accuracy = np.mean(all_number_accuracies)  # 计算平均数字准确率
    
    # 将结果保存到文件
    result_file = "reconstruction_results.txt"
    with open(result_file, "w") as f:
        f.write("平均分数:\n")
        f.write(f"BLEU-1: {avg_bleu1:.4f}\n")
        f.write(f"BLEU-2: {avg_bleu2:.4f}\n")
        f.write(f"BLEU-3: {avg_bleu3:.4f}\n")
        f.write(f"BLEU-4: {avg_bleu4:.4f}\n")
        f.write(f"ROUGE-1: {avg_rouge1:.4f}\n")
        f.write(f"ROUGE-2: {avg_rouge2:.4f}\n")
        f.write(f"ROUGE-L: {avg_rougeL:.4f}\n\n")
        f.write(f"数字准确率: {avg_number_accuracy:.4f}\n\n")  # 添加数字准确率结果
        
        f.write("详细结果:\n")
        for i in range(len(all_sources)):
            f.write(f"样本 {i}:\n")
            f.write(f"原文: {all_sources[i]}...\n")  # 只保存前200个字符
            f.write(f"预测: {all_predictions[i]}...\n")
            f.write(f"BLEU-1: {all_bleu1[i]:.4f}, BLEU-2: {all_bleu2[i]:.4f}, ")
            f.write(f"BLEU-3: {all_bleu3[i]:.4f}, BLEU-4: {all_bleu4[i]:.4f}\n")
            f.write(f"ROUGE-1: {all_rouge1[i]:.4f}, ROUGE-2: {all_rouge2[i]:.4f}, ROUGE-L: {all_rougeL[i]:.4f}\n\n")
            f.write(f"数字准确率: {all_number_accuracies[i]:.4f}\n\n")  # 添加每个样本的数字准确率
    
    # 输出结果
    print("\n======= 评估结果 =======")
    print(f"平均 BLEU-1: {avg_bleu1:.4f}")
    print(f"平均 BLEU-2: {avg_bleu2:.4f}")
    print(f"平均 BLEU-3: {avg_bleu3:.4f}")
    print(f"平均 BLEU-4: {avg_bleu4:.4f}")
    print(f"平均 ROUGE-1: {avg_rouge1:.4f}")
    print(f"平均 ROUGE-2: {avg_rouge2:.4f}")
    print(f"平均 ROUGE-L: {avg_rougeL:.4f}")
    print(f"平均 数字准确率: {avg_number_accuracy:.4f}")  # 添加平均数字准确率输出
    print(f"结果已保存到 {result_file}")
    
    return {
        'bleu1': avg_bleu1,
        'bleu2': avg_bleu2,
        'bleu3': avg_bleu3,
        'bleu4': avg_bleu4,
        'rouge1': avg_rouge1,
        'rouge2': avg_rouge2,
        'rougeL': avg_rougeL,
        'number_accuracy': avg_number_accuracy,  # 添加数字准确率到返回结果
    }

if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.lm_path,
        model_max_length=8192,
        use_fast=True,
        trust_remote_code=True
    )
    
    data_dir = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/ehr_free_text/reconstruction_mini_20.jsonl"
    data = MIMICDataset(data_dir)

    icformer = ICFormerModel.from_pretrained(args.icformer_path, device_map="cuda", torch_dtype=torch.bfloat16)
    icformer.requires_grad_(False)
    language_model = AutoModelForCausalLM.from_pretrained(args.lm_path, device_map='cuda', torch_dtype=torch.bfloat16, trust_remote_code=True)
    language_model.requires_grad_(False)

    model = ICFormer(icformer, language_model, tokenizer)
    model.max_seq_len = args.max_seq_len
    model.max_chunk_len = args.max_chunk_len
    model.use_chunk = args.use_chunk

    ckpt = torch.load(os.path.join(args.icformer_path, 'param.pt'))
    with torch.no_grad():
        model.digest_embeddings.copy_(ckpt['digest_embeddings'])
        model.AE.copy_(ckpt['AE'])

    # 评估整个数据集
    # 可以通过设置num_samples参数来控制评估的样本数量
    results = evaluate_dataset(model, data, num_samples=3, max_new_tokens=2048)
        