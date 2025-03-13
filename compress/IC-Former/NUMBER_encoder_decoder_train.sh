CUDA_VISIBLE_DEVICES=0 python NUMBER_encoder_decoder_train.py \
    --data_path /Users/tangbohao/Files/科研资料/EHR-decision/mimic-code/ehr_freetext/reconstruction_256len_mini.jsonl \
    --save_dir ./number_module_output \
    --epochs 20000 \
    --batch_size 64 \
    --lr 0.001 \
    --hidden_dim 512 \
    --output_dim 3584 \
    --save_interval 10 \
    --log_example_interval 1 # 每 5 个 epoch 输出验证样本
