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
    model_name_or_path: Optional[str] = field(default= "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/bohao_mimic_project/SFT/results/trail_1/checkpoint-844")
    model_name: Optional[str] = field(default= "trial_1.8B")
    is_lora: Optional[bool] = field(default=False)
    lora_rank: Optional[int] = field(default=16)
    target_modules :Optional[List[str]] = field(default=None)
    random_flag: Optional[str] = field(default=True)


@dataclass
class DataArguments:
    patient_root_dir: str = field(default="/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/my_MIMIC_IV/patients_sorted/", metadata={"help": "Path to the training data."})
    patient_id_csv: str = field(default="/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/bohao_mimic_project/supplementary_files/new_split/test_shuffle.csv", metadata={"help": "Path to the training data."})


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
    test_dataset = MIMIC_Dataset(patient_root_dir = data_args.patient_root_dir, patient_id_csv = data_args.patient_id_csv, tokenizer=tokenizer)
    
    # with open(training_args.output_dir + '/' + model_args.model_name + '.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['question','answer','prediction'])
        
    #     for index in tqdm.tqdm(range(min(len(test_dataset),1500))):
    #         sample = test_dataset.get_infer_case(index, model_args.random_flag)
    #         question = sample["question"]
    #         answer = sample["answer"]
    #         question_ids = tokenizer(
    #             question,
    #             return_tensors="pt", 
    #             add_special_tokens=False,
    #             max_length=tokenizer.model_max_length-100,
    #         ).to("cuda")
    #         with torch.no_grad():
    #             generated = model.generate(**question_ids, max_new_tokens=100, do_sample=False, top_k=50)
    #             # print(generated)
    #             # print(question_ids['input_ids'])
    #             # print(generated[0][len(question_ids['input_ids'][0]):])
    #             prediction = tokenizer.decode(generated[0][len(question_ids['input_ids'][0]):],False)
    #         writer.writerow([question,answer,prediction])
    #         # break
    with open(training_args.output_dir + '/' + model_args.model_name + '_' +model_args.random_flag + '.json', 'w') as f:
        data = []
        if model_args.random_flag in ["1", "2", "4", "5"]:
            # 从test_dataset中取balance的样本，answer = “yes”和“no” 各取50个
            print("单选")
            test_list = []
            yes_num = 0
            no_num = 0

            # 创建两个进度条
            yes_pbar = tqdm.tqdm(total=50, desc='Collecting YES samples', position=0)
            no_pbar = tqdm.tqdm(total=50, desc='Collecting NO samples', position=1)

            for index in tqdm.tqdm(range(len(test_dataset))):
                _, _, task = test_dataset.samples[index]
                if task != model_args.random_flag:
                    continue
                sample = test_dataset.get_infer_case(index)
                # 如果answer是yes字符串开头
                if sample["answer"].startswith("yes"):
                    if yes_num < 50:
                        test_list.append(sample)
                        yes_num += 1
                        yes_pbar.update(1)  # 更新yes进度条
                elif sample["answer"].startswith("no"):
                    if no_num < 50:
                        test_list.append(sample)
                        no_num += 1
                        no_pbar.update(1)  # 更新no进度条
                if yes_num == 50 and no_num == 50:
                    print("\nBalance finished successfully")
                    break
            # 关闭进度条
            yes_pbar.close()
            no_pbar.close()
            if yes_num < 50 or no_num < 50:
                print("\nBalance not finished - insufficient samples")
                return

            for index in tqdm.tqdm(range(len(test_list))):
                sample = test_list[index]
                question = sample["question"]
                answer = sample["answer"]
                question_ids = tokenizer(
                    question,
                    return_tensors="pt", 
                    add_special_tokens=False,
                    max_length=tokenizer.model_max_length-100,
                ).to("cuda")
                with torch.no_grad():
                    generated = model.generate(**question_ids, max_new_tokens=100, do_sample=False, top_k=50)
                    prediction = tokenizer.decode(generated[0][len(question_ids['input_ids'][0]):],False)
                    # outputs = model(**question_ids)
                    # logits = outputs.logits
                    # # print(logits[0].shape)
                    
                    # # 获取所有位置的最大概率token ids
                    # predicted_token_ids = torch.argmax(logits[0], dim=-1)
                    
                    # # 将所有token ids转换为tokens
                    # predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_token_ids)
                    
                    # # 转换为完整文本
                    # prediction = tokenizer.convert_tokens_to_string(predicted_tokens)
                data.append({"question": question,"answer": answer,"prediction": prediction})
        else:
            print("多选")
            test_list = []
            yes_num = 0
            yes_pbar = tqdm.tqdm(total=100, desc='Collecting samples', position=0)
            for index in tqdm.tqdm(range(len(test_dataset))):
                _, _, task = test_dataset.samples[index]
                if task != model_args.random_flag:
                    continue
                sample = test_dataset.get_infer_case(index)
                # 如果answer是yes字符串开头
                
                test_list.append(sample)
                yes_num += 1
                yes_pbar.update(1)  # 更新yes进度条
                
                if yes_num == 100:
                    print("\Collect finished successfully")
                    break
            # 关闭进度条
            yes_pbar.close()
            for index in tqdm.tqdm(range(len(test_list))):
                sample = test_list[index]
                question = sample["question"]
                answer = sample["answer"]
                question_ids = tokenizer(
                    question,
                    return_tensors="pt", 
                    # add_special_tokens=False,
                    truncation=False,
                    # max_length=tokenizer.model_max_length-100,
                ).to("cuda")
                # print(question_ids)
                # input_ids = question_ids["input_ids"]
                # predicted_tokens = tokenizer.convert_ids_to_tokens(torch.tensor([281]))
                # print(predicted_tokens)
                # with open("file22.json", "w") as f:
                #     json.dump({"input_ids": input_ids.tolist()}, f)
                # exit()
                with torch.no_grad():
                    generated = model.generate(**question_ids, max_new_tokens=100, do_sample=False, top_k=50)
                    prediction = tokenizer.decode(generated[0][len(question_ids['input_ids'][0]):],False)
                    # print(question_ids)
                    
                    # outputs = model(**question_ids)
                    
                    # logits = outputs.logits
                    # # print(logits[0].shape)
                    
                    # # 获取所有位置的最大概率token ids
                    # predicted_token_ids = torch.argmax(logits[0], dim=-1)
                    # print(predicted_token_ids)
                    # exit()
                    # # 将所有token ids转换为tokens
                    # predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_token_ids)
                    
                    # # 转换为完整文本
                    # prediction = tokenizer.convert_tokens_to_string(predicted_tokens)
                data.append({"question": question,"answer": answer,"prediction": prediction})
        json.dump(data, f, ensure_ascii=False, indent=2)
            


if __name__ == "__main__":
    train()