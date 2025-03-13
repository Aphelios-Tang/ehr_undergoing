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
from my_mimic_dataset_hadm_id_cutlen_single_seeno import MIMIC_Dataset
# from peft import LoraConfig, get_peft_model
from typing import Dict, Optional, Sequence, List

from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter

class TokenLossCallback(TrainerCallback):
    def __init__(self, logging_dir):
        self.writer = SummaryWriter(log_dir=logging_dir)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "Token_Loss" in logs:
            self.writer.add_text("Token_Loss", logs["Token_Loss"], state.global_step)
        if logs and "Avg_Per_Token_Loss" in logs:
            self.writer.add_scalar("Avg_Per_Token_Loss", logs["Avg_Per_Token_Loss"], state.global_step)
    
    def on_train_end(self, args, state, control, **kwargs):
        self.writer.close()


SEED = 42
transformers.set_seed(SEED)
IGNORE_INDEX = -100
        
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    is_lora: Optional[bool] = field(default=False)
    lora_rank: Optional[int] = field(default=16)
    target_modules :Optional[List[str]] = field(default=None)


@dataclass
class DataArguments:
    patient_root_dir: str = field(default="/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/my_MIMIC_IV/patients_sorted/", metadata={"help": "Path to the training data."})
    patient_id_csv: str = field(default="/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/bohao_mimic_project/supplementary_files/split/train_patients.csv", metadata={"help": "Path to the training data."})


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
    fsdp: Optional[str] = field(default=None)
    fsdp_transformer_layer_cls_to_wrap: Optional[str] = field(default=None)

    ############################################
    logging_dir: Optional[str] = field(default="./logs")  # 日志保存目录
    logging_steps: int = field(default=10)  # 每隔多少步记录一次日志
    report_to: Optional[List[str]] = field(
        default_factory=lambda: ["tensorboard"]
    )  # 使用 tensorboard 进行可视化

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]):
        try:
            # print(instances)
            input_ids, labels, tokens_0 = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "tokens_0"))
            # # 提取 input_ids、labels 和 tokens
            # input_ids = [instance["input_ids"] for instance in instances]
            # labels = [instance["labels"] for instance in instances]
            # tokens_0 = [instance["tokens_0"] for instance in instances]
        except KeyError as e:
            missing_key = e.args[0]
            raise KeyError(f"Missing key '{missing_key}' in some instances. 请确保每个样本包含 '{missing_key}' 字段。")

        # 填充 input_ids 和 labels
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        
        # 确定批次中的最大序列长度
        max_length = input_ids_padded.size(1)
        
        # 填充 tokens，使其长度与 input_ids 对齐
        tokens_padded = [
            token + [""] * (max_length - len(token)) for token in tokens_0
        ]
        
        return dict(
            input_ids=input_ids_padded,
            labels=labels_padded,
            attention_mask=input_ids_padded.ne(self.tokenizer.pad_token_id),
            tokens_0=tokens_padded  # 添加 tokens
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = MIMIC_Dataset(patient_root_dir = data_args.patient_root_dir, patient_id_csv = data_args.patient_id_csv, tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        logits = outputs.logits
        labels = inputs.get("labels")
        tokens_0 = inputs.get("tokens_0")  # 获取 tokens 列表

        if tokens_0 is None:
            raise ValueError("输入的 batch 中缺少 'tokens' 字段。请确保 Dataset 和 DataCollator 正确传递 'tokens'。")

        # 计算每个 token 的 loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=IGNORE_INDEX)
        per_token_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        per_token_loss = per_token_loss.view(labels.size())  # 形状: (batch_size, sequence_length)

        if self.state.global_step % self.args.logging_steps == 0:
            # 获取当前 batch 的第一个样本
            first_sample_loss = per_token_loss[0].detach().cpu().numpy()
            first_sample_tokens = tokens_0[0]
            
            # 将 tokens 和对应的 loss 拼接成字符串，忽略填充的空字符串
            token_loss_pairs = [
                f"{token}: {loss_item:.4f}" 
                for token, loss_item in zip(first_sample_tokens, first_sample_loss) if token
            ]
            token_loss_str = ' | '.join(token_loss_pairs)
            
            # 使用 self.log 记录文本信息和平均损失
            self.log({
                "Token_Loss": token_loss_str,
                "Avg_Per_Token_Loss": float(first_sample_loss.mean())
            })

        return (loss, outputs) if return_outputs else loss

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        use_fast=False,
        trust_remote_code=True
    )
            
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = CustomTrainer(model=model, tokenizer=tokenizer, args=training_args, callbacks=[TokenLossCallback(training_args.logging_dir)], **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()