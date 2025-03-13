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


SEED = 42
transformers.set_seed(SEED)
IGNORE_INDEX = -100

# # Custom Trainer class to handle FSDP saving
# class CustomTrainer(Trainer):
#     def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
#         """
#         Save the model. Overrides Trainer's save_model to handle FSDP.
#         """
#         if self.fsdp is not None:
#             output_dir = output_dir or self.args.output_dir
#             save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
#             with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
#                 cpu_state_dict = self.model.state_dict()
#             if self.args.should_save:
#                 self._save(output_dir, state_dict=cpu_state_dict)
#             if self.args.push_to_hub and not _internal_call:
#                 self.push_to_hub(commit_message="Model save")
#         else:
#             super().save_model(output_dir, _internal_call)

# # Utility function to handle safe saving of models for Hugging Face trainers
# def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
#     """
#     Safely saves the model state for Hugging Face trainers with custom handling for FSDP.
#     """
#     save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
#     with FSDP.state_dict_type(trainer.model, StateDictType.FULL_STATE_DICT, save_policy):
#         cpu_state_dict = trainer.model.state_dict()
#     if trainer.args.should_save:
#         trainer._save(output_dir, state_dict=cpu_state_dict)
        
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    is_lora: Optional[bool] = field(default=True)
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

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = MIMIC_Dataset(patient_root_dir = data_args.patient_root_dir, patient_id_csv = data_args.patient_id_csv, tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


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
    
    # if model_args.is_lora:
    #     if model_args.target_modules is None:
    #         raise ValueError("target_modules is required for LoRA")
        
    #     config = LoraConfig(
    #         r = model_args.lora_rank,
    #         lora_alpha = model_args.lora_alpha,
    #         target_modules = model_args.target_modules,
    #         lora_dropout = 0.05,
    #         bias = 'none',
    #         task_type="CAUSAL_LM",
    #     )
        
    #     model = get_peft_model(model, config)
            
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()