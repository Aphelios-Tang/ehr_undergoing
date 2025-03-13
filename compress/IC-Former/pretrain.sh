CUDA_VISIBLE_DEVICES=0,1 python pretrain_ehr.py \
--lm_path /inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/mmedlm_model/MMedLM2-1_8B/MMedLM2-1_8B \
--save_path ./output \
--gradient_accumulation 32 \
--max_seq_len 512 \
--max_chunk_len 512 \
--num_query_tokens 128 \
--save_optimizer \
--clip_grad \
--max_epoch 200 \
--num_hidden_layers 5 \
--shuffle 

