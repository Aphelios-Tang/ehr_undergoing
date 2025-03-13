source /inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/miniconda3/bin/activate
conda activate mimic
torchrun --nproc_per_node=2 --master_port 17999 train_from_freetext.py \
    --model_name_or_path /inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/mmedlm_model/MMedLM2-1_8B/MMedLM2-1_8B \
    --output_dir /inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/bohao_mimic_project/SFT/results/try \
    --bf16 True \
    --num_train_epochs 100 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 1e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --logging_dir "./logs_fenbushi" \
    --fsdp "full_shard auto_wrap" \
    --run_name "mimic_sequential_trial" \
    --gradient_checkpointing True \
    --gradient_clipping 1.0

# --fsdp "full_shard auto_wrap" \
# --fsdp_transformer_layer_cls_to_wrap TRANSFORMER_LAYER \