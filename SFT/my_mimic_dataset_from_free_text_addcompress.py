from torch.utils.data import Dataset
from functools import *
from utils import * # 包含数据处理的函数，xxx_item_to_free_text
import transformers
import copy
from DS import *
import torch

SEED = 42
transformers.set_seed(SEED)
IGNORE_INDEX = -100



def preprocess(sample, tokenizer, icformer_model, language_model):
    """
    处理样本，将inputs_list分批通过icformer生成soft prompt，与instruction和output的embeds拼接。
    返回包含inputs_embeds、labels和attention_mask的字典。
    """
    # sample: {"input": ["xxx", "xxx", ...], "instruction": "xxx", "output": "xxx", "task_id", "xxx"}
    inputs_list, instruction, output, task_id = sample["input"], sample["instruction"], sample["output"], sample["task_id"]
 
    embedding_layer = language_model.get_input_embeddings()
    device = language_model.device
    max_seq_len = icformer_model.max_seq_len

    # 1. 手动分批处理inputs_list
    batches = []
    current_batch = []
    current_len = 0

    for text in inputs_list:
        text_ids = tokenizer(text, return_tensors="pt")["input_ids"][0] # torch.Size([n]),包含bos
        text_len = len(text_ids)

        if(current_len + text_len > max_seq_len):
            if(current_batch):
                batches.append(current_batch)
                current_batch = []
                current_len = 0
        current_batch.append(text)
        current_len += text_len
    
    if(current_batch):
        batches.append(current_batch)
    
    # 2. 每个batch生成soft prompt
    soft_prompts = []
    for batch in batches:
        batch_text = "\n".join(batch)
        # batch_text = tokenizer.bos_token + batch_text
        batch_ids = tokenizer(batch_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
        # batch_ids = batch_ids.contiguous()
        # batch_ids: torch.Size([batch_len])
        batch_embeds = embedding_layer(batch_ids)  # 形状：(batch_len, hidden_size)
        batch_embeds = batch_embeds.unsqueeze(0)  # 形状：(1, batch_len, hidden_size)

        soft_prompt = icformer_model.get_soft_prompt(inputs_embeds=batch_embeds, use_chunk=icformer_model.use_chunk) # torch.Size([1, 128, 2048])
    
        soft_prompts.append(soft_prompt) 
    
    # 3. 拼接soft prompt
    soft_prompt = torch.cat(soft_prompts, dim=1)

    # 4. 处理instruction和output
    instruction_ids = tokenizer(instruction, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
    output_ids = tokenizer(output, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)

    instruction_embeds = embedding_layer(instruction_ids).unsqueeze(0)  # 形状：(1, instr_len, hidden_size)
    output_embeds = embedding_layer(output_ids).unsqueeze(0)  # 形状：(1, out_len, hidden_size)

    # 4.1. 添加pre_embeds和post_embeds
    pre_embeds = embedding_layer(tokenizer("<s>[INST] Response the Prompt based on the below text:\n\n")).unsqueeze(0)
    post_embeds = embedding_layer(tokenizer("[/INST]")).unsqueeze(0)

    # 5. 拼接 soft prompt, instruction 和 output
    inputs_embeds = torch.cat([pre_embeds, soft_prompt, icformer_model.FT, instruction_embeds, post_embeds, output_embeds], dim=1)  # 形状：(1, total_len, hidden_size)

    # 6. 生成labels
    soft_prompt_len = soft_prompt.size(1)
    instruction_len = instruction_embeds.size(1)
    output_len = output_embeds.size(1)
    total_len = soft_prompt_len + instruction_len + output_len

    labels = torch.full((1, total_len), IGNORE_INDEX, dtype=torch.long, device=device)  # 非预测部分设为 -100
    labels[0, soft_prompt_len + instruction_len:] = output_ids  # 只预测 output 部分
    
    # 7. 生成 attention_mask
    attention_mask = torch.ones((1, total_len), dtype=torch.long, device=device)

    return {
        "inputs_embeds": inputs_embeds.squeeze(0),  # 形状：(total_len, hidden_size)
        "labels": labels.squeeze(0),  # 形状：(total_len)
        "attention_mask": attention_mask.squeeze(0)  # 形状：(total_len)
    }


class MIMIC_Dataset(Dataset):
    def __init__(self, dataset_json, tokenizer, icformer_model, language_model):
        self.samples = read_jsonl(dataset_json)
        self.tokenizer = tokenizer
        self.icformer_model = icformer_model
        self.language_model = language_model

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,idx):
        sample = self.samples[idx]
        tokens = preprocess(sample, self.tokenizer, self.icformer_model, self.language_model)
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

