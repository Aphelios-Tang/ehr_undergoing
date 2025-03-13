import torch
import os
import transformers
from random import shuffle
from transformers import Trainer
from dataclasses import dataclass, field
from my_mimic_dataset_from_free_text_addcompress import MIMIC_Dataset
from typing import Dict, Optional, Sequence, List
from IC import modules
from IC.modules import ICFormer, ICFormerModel



SEED = 42
transformers.set_seed(SEED)
IGNORE_INDEX = -100

        
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    icformer_path: Optional[str] = field(default=None)
    is_lora: Optional[bool] = field(default=False)
    lora_rank: Optional[int] = field(default=16)
    target_modules :Optional[List[str]] = field(default=None)


@dataclass
class DataArguments:
    dataset_json: str = field(default="/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/ehr_free_text/train.jsonl", metadata={"help": "Path to the training data."})

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
        # input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        inputs_embeds, labels, attention_masks = tuple(
            [instance[key] for instance in instances] for key in ("inputs_embeds", "labels", "attention_mask")
        )
        # input_ids = torch.nn.utils.rnn.pad_sequence(
        #     input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        # )
        # 对inputs_embeds进行padding
        inputs_embeds = torch.nn.utils.rnn.pad_sequence(
            inputs_embeds, batch_first=True, padding_value=0.0  # embeds是浮点数，padding值为0.0
        )
        # 对labels进行padding
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        # 对attention_mask进行padding
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )

        return dict(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_masks,
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, icformer, language_model) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = MIMIC_Dataset(dataset_json = data_args.dataset_json, tokenizer=tokenizer, icformer_model=icformer, language_model=language_model)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    llm_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=8192,
        use_fast=False,
        trust_remote_code=True
    )

    icformer = ICFormer(
        icformer=ICFormerModel.from_pretrained(model_args.icformer_path, device_map="cuda", torch_dtype=torch.bfloat16),
        language_model=llm_model,
        tokenizer = tokenizer
    )
    ckpt = torch.load(os.path.join(model_args.icformer_path, 'param.pt'))
    with torch.no_grad():
        icformer.digest_embeddings.copy_(ckpt['digest_embeddings'])
    
    # icformer不参与训练
    # icformer.requires_grad_(False)
    icformer.icformer.requires_grad_(False)
    icformer.AE.requires_grad_(False)
    # 随机初始化icformer的FT
    icformer.FT.requires_grad_(True)
    

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, icformer=icformer, language_model=llm_model)

    trainer = Trainer(model=llm_model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()