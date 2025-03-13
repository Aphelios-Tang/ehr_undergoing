#!/bin/bash
#SBATCH -J mimic_sequential_trial
#SBATCH -p medai        # 指定分区
#SBATCH --nodes=1            # 节点数
#SBATCH --ntasks=1           # 任务数
#SBATCH --gres=gpu:8       # 指定GPU资源
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=24G
#SBATCH --quotatype=auto

srun torchrun --nproc_per_node=8 --master_port 17999 train.py \
    --model_name_or_path /mnt/petrelfs/wuchaoyi/LLMModels/Model/MMedLM2-1_8B \
    --output_dir /mnt/petrelfs/wuchaoyi/my_MIMIC_IV_2.2/SFT/results/trial_2 \
    --bf16 True \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 1e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --run_name "mimic_sequential_trial" \
    --gradient_checkpointing True \
    --gradient_clipping 1.0

# --fsdp "full_shard auto_wrap" \
# --fsdp_transformer_layer_cls_to_wrap TRANSFORMER_LAYER \