CUDA_VISIBLE_DEVICES=0 python NUMBER_pretrain_ehr.py \
--lm_path /inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/mmedlm_model/Qwen2.5-7B-Instruct \
--save_path ./NUMBER_output_qwen_7b_max_layer5_ae32_512-128_text1024cut_number \
--gradient_accumulation 32 \
--max_seq_len 512 \
--num_query_tokens 128 \
--save_optimizer \
--clip_grad \
--max_epoch 500 \
--num_hidden_layers 5 \
--save_interval 32000 \
--eval_interval 20 \
--eval_samples 2 \
--lr 0.0001 \
--shuffle 

