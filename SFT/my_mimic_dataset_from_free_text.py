import os 
import pandas as pd
import json
import datetime
import jsonlines
from torch.utils.data import Dataset, DataLoader
import tqdm
from functools import *
import random
from utils import * # 包含数据处理的函数，xxx_item_to_free_text
import transformers
import copy
from DS import *
import torch

SEED = 42
transformers.set_seed(SEED)
IGNORE_INDEX = -100

def preprocess(sample, tokenizer):
    # sample: {"input": ["xxx", "xxx", ...], "instruction": "xxx", "output": "xxx", "task_id", "xxx"}
    inputs_list, instruction, output, task_id = sample["input"], sample["instruction"], sample["output"], sample["task_id"]
    inputs = "\n".join(inputs_list)
    
    sources = tokenizer.bos_token + inputs + SPLIT_LINE + instruction
    targets = output
    
    sources_tokenized = tokenizer(sources, return_tensors="pt", truncation=False)
    sources_ids = sources_tokenized["input_ids"][0]
    
    target_tokenized = tokenizer(targets, return_tensors="pt", truncation=False)
    target_ids = target_tokenized["input_ids"][0]

    combined_ids = torch.cat([sources_ids, target_ids[1:]], dim=0)
    input_ids = combined_ids

    labels = copy.deepcopy(input_ids)

    labels[:len(sources_tokenized["input_ids"][0])] = IGNORE_INDEX

    if len(input_ids) > tokenizer.model_max_length:
        input_ids = input_ids[-tokenizer.model_max_length:]
        labels = labels[-tokenizer.model_max_length:]

    return dict(input_ids=input_ids, labels=labels)


class MIMIC_Dataset(Dataset):
    def __init__(self, dataset_json, tokenizer):
        self.samples = read_jsonl(dataset_json)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,idx):
        sample = self.samples[idx]
        tokens = preprocess(sample, self.tokenizer)
        return tokens
    
    def get_task_id(self, idx):
        return self.samples[idx]["task_id"]

    def get_infer_case(self, idx):
        sample = self.samples[idx]
        inputs_list, instruction, output, task_id = sample["input"], sample["instruction"], sample["output"], sample["task_id"]
        return {
            "inputs": inputs_list,
            "instruction": instruction,
            "output": output,
            "task_id": task_id
        } 
    

if __name__ == "__main__":

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/mmedlm_model/MMedLM2-1_8B/MMedLM2-1_8B",
        model_max_length=8192,
        use_fast=False,
        trust_remote_code=True
    )
    dataset = MIMIC_Dataset(dataset_json = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/ehr_free_text/train_mini_4400.jsonl",tokenizer= tokenizer)
    sample = dataset.get_infer_case(0)
    print(sample)

