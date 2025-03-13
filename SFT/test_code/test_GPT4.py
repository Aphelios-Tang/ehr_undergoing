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
from my_mimic_dataset_hadm_id_cutlen import MIMIC_Dataset
# from peft import LoraConfig, get_peft_model
from typing import Dict, Optional, Sequence, List
import csv
import tqdm
from openai import OpenAI
import requests
from requests.exceptions import ProxyError, ConnectionError

SEED = 42
transformers.set_seed(SEED)
IGNORE_INDEX = -100

def send_request_gpt(messages, client):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages
    )
    return response.choices[0].message.content

def send_request_claude(messages, proxy_url, headers):
    """Sends a request to the model and returns the response content."""
    payload = json.dumps({
        "model": "gpt-4-turbo",
        "messages": messages
    })
    response = requests.post(proxy_url, headers=headers, data=payload)
    response_content = response.json()
    try:
        return response_content['choices'][0]['message']['content']
    except:
        print(response_content)
        
class GPT_Chatbot:
    def __init__(self, chatbot_type = "GPT-4", model_path_or_api = "sk-tPphc4JsztVS3EnOSFYcTMEu4VTD5lS8iqFBTqS2gOTkNuwG"):
        
        Baseurl = "https://api.claudeshop.top"
        Skey = "sk-YKIsf5FHQhY3NdQW9f6944Dd2d5742049d7cDb398e66B862"

        self.proxy_url = Baseurl + "/v1/chat/completions"
        self.headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {Skey}',
                'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                'Content-Type': 'application/json'
            }
        
    def chat_function(self, messages):
        """
        input: messages 
            messages = [
                {"role": "system", "content": initial_prompt},
                {"role": "user", "content": query}
            ]
            messages.extend([
                {"role": "assistant", "content": f"{value_dict}"},
                {"role": "user", "content": decomposition_prompt},
                {"role": "assistant", "content": decomposition_answer}
            ])
        return str
        """
        
        output = send_request_claude(messages, self.proxy_url, self.headers)
        return output


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default= "sk-7rJ2cgxeqYnq2AcVKuUQFb8ydvxwFLZE2fcCredPBwATZXgv")
    model_name: Optional[str] = field(default= "GPT4")
    is_lora: Optional[bool] = field(default=False)
    lora_rank: Optional[int] = field(default=16)
    target_modules :Optional[List[str]] = field(default=None)
    random_flag: Optional[str] = field(default=True)


@dataclass
class DataArguments:
    patient_root_dir: str = field(default="/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/my_MIMIC_IV/patients_sorted/", metadata={"help": "Path to the training data."})
    patient_id_csv: str = field(default="/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/bohao_mimic_project/supplementary_files/split/test_patients.csv", metadata={"help": "Path to the training data."})


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
    output_dir: str = field(default="/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/bohao_mimic_project/SFT/predic_csv")

# @dataclass
# class DataCollatorForSupervisedDataset(object):
#     """Collate examples for supervised fine-tuning."""

#     tokenizer: transformers.PreTrainedTokenizer

#     def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
#         input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
#         input_ids = torch.nn.utils.rnn.pad_sequence(
#             input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
#         )
#         labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
#         return dict(
#             input_ids=input_ids,
#             labels=labels,
#             attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
#         )


# def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
#     """Make dataset and collator for supervised fine-tuning."""
#     train_dataset = MIMIC_Dataset(patient_root_dir = data_args.patient_root_dir, patient_id_csv = data_args.patient_id_csv, tokenizer=tokenizer)
#     data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
#     return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.model_name == "GPT4":
        model = GPT_Chatbot()
        
    test_dataset = MIMIC_Dataset(patient_root_dir = data_args.patient_root_dir, patient_id_csv = data_args.patient_id_csv, tokenizer=None)
    
    with open(training_args.output_dir + '/' + model_args.model_name + '_' +model_args.random_flag + '.json', 'w') as f:
        data = []
        for index in tqdm.tqdm(range(min(len(test_dataset),10))):
            sample = test_dataset.get_infer_case(index, model_args.random_flag)
            question = sample["question"]
            answer = sample["answer"]
            # 告诉他严格按照要求的格式回答，不需要分析，只给出答案
            messages = [
                {"role": "system", "content": "You are a helpful assistant. You need to strictly follow the format of the answer and do not need to analyze the question. Please give the answer directly."},
                {"role": "user", "content": question}
            ]
            for i in range(10):
                try:
                    prediction = model.chat_function(messages)
                    break
                except:
                    pass
            data.append({"question": question,"answer": answer,"prediction": prediction})
        json.dump(data, f, ensure_ascii=False, indent=2) 


if __name__ == "__main__":
    train()