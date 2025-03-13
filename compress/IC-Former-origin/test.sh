CUDA_VISIBLE_DEVICES=0 python test_pretrain.py \
--lm_path /inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/mmedlm_model/llama2-7b-chat/Llama-2-7b-chat-hf \
--save_path ./output_7b \
--gradient_accumulation 32 \
--max_seq_len 512 \
--num_query_tokens 128 \
--save_optimizer \
--clip_grad \
--max_epoch 200 \
--num_hidden_layers 3 \
--icformer_path /inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/longcontext/model/IC-Former/pretrain \
--shuffle 

