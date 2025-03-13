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
from joblib import Parallel, delayed
import torch

SEED = 42
transformers.set_seed(SEED)
IGNORE_INDEX = -100

class MIMICDataset(Dataset):
    def __init__(self, jsonl_file_path):
        self.data = []
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())   
                self.data.append(item['input'])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]            

def process_item(idx, dataset):
    sample_out = dataset[idx]
    lines = []
    for text_block in sample_out:
        lines.append(json.dumps({"text": text_block}, ensure_ascii=False) + "\n")
    return lines

if __name__ == "__main__":
    dataset = MIMICDataset("/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/ehr_free_text/train_max.jsonl")

    output_file = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/ehr_free_text/reconstruction_item_max.jsonl"


    with open(output_file, "w", encoding="utf-8") as f:
        for data in tqdm.tqdm(dataset):
            lines = []
            for text_block in data:
                lines.append(json.dumps({"text": text_block}, ensure_ascii=False) + "\n")
            f.writelines(lines)

            
    
    
