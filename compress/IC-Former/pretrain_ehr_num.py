import os
import torch
import transformers
from transformers import AutoModelForCausalLM, LlamaTokenizer
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

from icformer import (
    ICFormerConfig, 
    ICFormerModel, 
)
from modules_num import Trainer, ICFormer
from data_utils import PileDataset, MIMICDataset
# from my_mimic_dataset_for_reconstruction import MIMIC_Dataset
from utils import parse_args, seed_everything

SEED = 42
transformers.set_seed(SEED)
IGNORE_INDEX = -100

if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    # data = PileDataset(args.data_path)
    

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.lm_path, trust_remote_code=True)

    # patient_root_dir = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/my_MIMIC_IV/patients_sorted/"
    # patient_id_csv = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/bohao_mimic_project/supplementary_files/new_split/train_mini.csv"
    
    data_dir = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/ehr_free_text/reconstruction_item_max.jsonl"

    # data = MIMIC_Dataset(patient_root_dir = patient_root_dir, patient_id_csv = patient_id_csv, tokenizer=tokenizer)
    data = MIMICDataset(data_dir)

    # tokenizer = LlamaTokenizer.from_pretrained(args.lm_path, use_fast=True)

    if args.icformer_path: # Load icformer checkpoint
        icformer = ICFormerModel.from_pretrained(args.icformer_path, device_map='cuda', torch_dtype=torch.bfloat16)
    else:                  # Random initialize icformer
        icformer_config = ICFormerConfig()
        icformer_config.num_hidden_layers = args.num_hidden_layers
        icformer_config.num_query_tokens = args.num_query_tokens
        icformer = ICFormerModel(icformer_config).to(dtype=torch.bfloat16, device='cuda')

    language_model = AutoModelForCausalLM.from_pretrained(args.lm_path, device_map='cuda', torch_dtype=torch.bfloat16)
    language_model.requires_grad_(False)

    model = ICFormer(icformer, language_model, tokenizer).to(torch.bfloat16)
    model.max_seq_len = args.max_seq_len
    model.max_chunk_len = args.max_chunk_len
    model.use_chunk = args.use_chunk
    # model.encode = args.encode

    if args.icformer_path: # Load digest embeddings and special tokens embeddings
        ckpt = torch.load(os.path.join(args.icformer_path, 'param.pt'))
        with torch.no_grad():
            model.digest_embeddings.copy_(ckpt['digest_embeddings'])
            model.AE.copy_(ckpt['AE'])
            model.A.copy_(ckpt['A'])
            model.B.copy_(ckpt['B'])
            model.C.copy_(ckpt['C'])
            model.D.copy_(ckpt['D'])
            if 'number_embedder_state_dict' in ckpt:
                model.number_embedder.load_state_dict(ckpt['number_embedder_state_dict'])
            if 'fusion_gate_state_dict' in ckpt:
                model.fusion_gate.load_state_dict(ckpt['fusion_gate_state_dict'])
    

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name}: {param.device}, {param.dtype}")
    # exit()   
        
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    if os.path.exists(os.path.join(args.icformer_path, 'optimizer.pt')):
        ckpt = torch.load(os.path.join(args.icformer_path, 'optimizer.pt'))
        optimizer.load_state_dict(ckpt)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)
    if os.path.exists(os.path.join(args.icformer_path, 'scheduler.pt')):
        ckpt = torch.load(os.path.join(args.icformer_path, 'scheduler.pt'))
        scheduler.load_state_dict(ckpt)

    trainer = Trainer(
        model=model,
        dataset=data,
        optimizer=optimizer,
        scheduler=None, # You can create your own scheduler
        max_epoch=args.max_epoch,
        save_interval=args.save_interval,
        save_dir=args.save_path,
        save_optimizer=args.save_optimizer,
        save_scheduler=args.save_scheduler,
        avg_level=args.avg_level,
        gradient_accumulation=args.gradient_accumulation,
        shuffle=args.shuffle,
        clip_grad=args.clip_grad,
        max_norm=args.max_norm,
        eval_interval=args.eval_interval if hasattr(args, 'eval_interval') else 5000,  # 默认5000步评估一次
        eval_samples=args.eval_samples if hasattr(args, 'eval_samples') else 5,  # 默认每次评估5个样本
    )

    trainer.train(
        start_epoch=0, # specify the last epoch if you want to resume training
        start_step=0,  # specify the last step(assuming dataset is not shuffled) if you want to resume training
    )

