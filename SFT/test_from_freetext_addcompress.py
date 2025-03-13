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
from my_mimic_dataset_from_free_text_addcompress import MIMIC_Dataset
# from peft import LoraConfig, get_peft_model
from typing import Dict, Optional, Sequence, List, Union
import csv
import tqdm
from IC import modules
from IC.modules import ICFormer, ICFormerModel


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
    model_name_or_path: Optional[str] = field(default= "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/bohao_mimic_project/SFT/results/try_comp/checkpoint-8112")
    icformer_path: Optional[str] = field(default="/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/longcontext/IC-Former/output_1_8b/checkpoint-36000")
    model_name: Optional[str] = field(default= "trial_1.8B")
    is_lora: Optional[bool] = field(default=False)
    lora_rank: Optional[int] = field(default=16)
    target_modules :Optional[List[str]] = field(default=None)
    random_flag: Optional[str] = field(default="1")


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
    output_dir: str = field(default="/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/bohao_mimic_project/SFT/predic_csv_freetext_compress")




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
    icformer = ICFormer(
        icformer=ICFormerModel.from_pretrained(model_args.icformer_path, device_map="cuda", torch_dtype=torch.bfloat16),
        language_model=model,
        tokenizer = tokenizer
    )
    ckpt = torch.load(os.path.join(model_args.icformer_path, 'param.pt'))
    with torch.no_grad():
        icformer.digest_embeddings.copy_(ckpt['digest_embeddings'])
    tokenizer.truncation_side='left'
    test_dataset = MIMIC_Dataset(dataset_json = data_args.dataset_json, tokenizer=tokenizer,  icformer_model=icformer, language_model=model)
    data = []
    with open(training_args.output_dir + '/' + model_args.model_name + '_' +model_args.random_flag + '.json', 'w') as f:
        for index in tqdm.tqdm(range(len(test_dataset))):
            sample = test_dataset.get_infer_case(index)
            if sample["task_id"] != model_args.random_flag:
                continue
            datas = test_dataset[index]
            inputs_embeds = datas["inputs_embeds"].unsqueeze(0).to(device)
            question = tokenizer.bos_token + "\n".join(sample["inputs"]) + "\n" + sample["instruction"]
            answer = sample["output"]
            
            with torch.no_grad():
                generated = model.generate(inputs_embeds = inputs_embeds, max_new_tokens=100, do_sample=False, top_k=50)
                
                prediction = tokenizer.decode(generated[0],False)
                
            data.append({"question": question,"answer": answer,"prediction": prediction})
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    train()