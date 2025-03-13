import os
import json
import copy
import torch
import logging
import transformers
from random import shuffle
from transformers import Trainer
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from my_mimic_dataset_from_free_text import MIMIC_Dataset
# from peft import LoraConfig, get_peft_model
from typing import Dict, Optional, Sequence, List, Union
import csv
import tqdm


SEED = 42
transformers.set_seed(SEED)
IGNORE_INDEX = -100

def parse_bool_or_str(value: str) -> Union[bool, str]:
    v = value.strip().lower()
    if v == True:
        return True
    elif v == False:
        return False
    return value
        
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default= "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/bohao_mimic_project/SFT/results/trail_1")
    model_name: Optional[str] = field(default= "trial_1.8B")
    is_lora: Optional[bool] = field(default=False)
    lora_rank: Optional[int] = field(default=16)
    target_modules :Optional[List[str]] = field(default=None)
    random_flag: Optional[str] = field(default=True)


@dataclass
class DataArguments:
    dataset_json: str = field(default="/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/ehr_free_text/test.jsonl", metadata={"help": "Path to the training data."})



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    use_cache : bool = field(default=False)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    gradient_clipping : float = field(
        default=None
    )
    output_dir: str = field(default="/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/bohao_mimic_project/SFT/predic_csv_freetext")




def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )
    device = torch.device("cuda")
    model = model.to(device)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        use_fast=False,
        trust_remote_code=True
    )
    tokenizer.truncation_side='left'
    test_dataset = MIMIC_Dataset(dataset_json = data_args.dataset_json, tokenizer=tokenizer)
    data = []
    with open(training_args.output_dir + '/' + model_args.model_name + '_' +model_args.random_flag + '.json', 'w') as f:
        for index in tqdm.tqdm(range(len(test_dataset))):
            sample = test_dataset.get_infer_case(index)
            if sample["task_id"] != model_args.random_flag:
                continue
            question = tokenizer.bos_token + "\n".join(sample["inputs"]) + "\n" + sample["instruction"]
            answer = sample["output"]
            question_ids = tokenizer(
                question,
                return_tensors="pt", 
                add_special_tokens=False,
                max_length=tokenizer.model_max_length-100,
            ).to("cuda")
            with torch.no_grad():
                generated = model.generate(**question_ids, max_new_tokens=100, do_sample=False, top_k=50)
                prediction = tokenizer.decode(generated[0][len(question_ids['input_ids'][0]):],False)
            data.append({"question": question,"answer": answer,"prediction": prediction})
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    train()