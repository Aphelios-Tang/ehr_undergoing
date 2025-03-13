import os
import gc
import math
import json
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
# from peft import PeftModel
# from utils import current_date_time
from .modeling_icformer import ICFormerModel

from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

class BaseModel(nn.Module):
    """
    A base class model suitable for training with Trainer, where all models inheriting from this class should at least implement 
    the train_step and get_pretrained_model methods.
    """
    def __init__(
        self,
        language_model:PreTrainedModel,
        tokenizer:PreTrainedTokenizer,
    ):
        super().__init__()
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.prepare_tokenizer(self.tokenizer)

    @staticmethod
    def prepare_tokenizer(tokenizer:PreTrainedTokenizer):
        tokenizer.pad_token_id = 0
        # bos has been added in the context
        tokenizer.add_bos_token = False
    
    @torch.no_grad()
    def generate(
        self,
        inputs_embeds:torch.Tensor,
        max_new_tokens:int=256,
        skip_special_tokens:bool=True,
        streaming:bool=False,
        return_output:bool=True,
        **kwargs,
    ):
        # Greedy decoding
        inputs_embeds = inputs_embeds.to(dtype=self.language_model.dtype)
        past_key_values = None
        output_ids = []
        for _ in range(max_new_tokens):
            output = self.language_model(inputs_embeds=inputs_embeds, past_key_values=past_key_values, use_cache=True)
            next_token_id = output.logits[0][-1].argmax(dim=-1)
            output_ids.append(next_token_id)
            if streaming:
                response = self.tokenizer.decode(output_ids)
                if not response: pass
                elif response[-1] == "\n": print()
                print(response.split('\n')[-1], end='\r', flush=True)
            if next_token_id == self.tokenizer.eos_token_id:
                break
            past_key_values = output.past_key_values
            next_embeds = self.language_model.get_input_embeddings()(torch.tensor([[next_token_id]], device=self.language_model.device))
            inputs_embeds = next_embeds
        if return_output:
            outputs = self.tokenizer.decode(output_ids, skip_special_tokens=skip_special_tokens, **kwargs)
            outputs = outputs.strip()
            return outputs, output_ids
        return None

    @torch.no_grad()
    def convert_ids_to_embeds(self, input_ids):
        if isinstance(input_ids, list):
            if isinstance(input_ids[0], list):
                input_ids = torch.tensor(input_ids)
            else:
                input_ids = torch.tensor([input_ids])
        input_ids = input_ids.to(device=self.language_model.device, dtype=torch.long)
        embeddings = self.language_model.get_input_embeddings()(input_ids)
        return embeddings
    
    def get_pretrained_model(self):
        raise NotImplementedError("get_pretrained_model method is not implemented.")
    
    def train_step(self, step, data):
        raise NotImplementedError("train_step method is not implemented.")
    
    def get_soft_prompt(self):
        raise NotImplementedError("get_soft_prompt method is not implemented.")


class ICFormer(BaseModel):
    def __init__(
        self, 
        icformer:ICFormerModel, 
        language_model:PreTrainedModel, 
        tokenizer:PreTrainedTokenizer
    ):
        super().__init__(language_model=language_model, tokenizer=tokenizer)
        self.icformer = icformer

        self.digest_embeddings = nn.Parameter(torch.zeros([1, icformer.config.num_query_tokens, icformer.config.hidden_size], device=icformer.device, dtype=icformer.dtype))
        self.AE = nn.Parameter(torch.zeros([1, 1, icformer.config.hidden_size], device=language_model.device, dtype=language_model.dtype))
        self.FT = nn.Parameter(torch.zeros([1, 1, icformer.config.hidden_size], device=language_model.device, dtype=language_model.dtype))
        
        self.max_seq_len = 512
        self.max_chunk_len = 512
        self.use_chunk = True
        self.encode = False


    def get_soft_prompt(
        self, 
        query_embeds=None, 
        input_ids=None, 
        inputs_embeds=None, 
        use_chunk=False,
        **kwargs,
    ):
        """
        Implement the soft prompt generation method.
        'use_chunk' is a boolean value to specify whether to apply divide-and-conquer strategy.
        """
        if query_embeds is None:
            query_embeds = self.digest_embeddings
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if input_ids is not None:
            inputs_embeds = self.convert_ids_to_embeds(input_ids)
        embeddings = inputs_embeds.to(device=self.icformer.device, dtype=self.icformer.dtype)

        # Causal attention mask for query tokens.
        query_mask = torch.tril(torch.ones([1, query_embeds.shape[1], query_embeds.shape[1]]))
        
        if not use_chunk:
            cross_attn_mask = torch.ones([1, query_embeds.shape[1], embeddings.shape[1]])
            attention_mask = torch.cat([cross_attn_mask, query_mask], dim=-1).to(device=self.icformer.device)
            hidden_states = torch.cat([embeddings, query_embeds], dim=1)
            soft_prompt = self.icformer(
                query_embeds=query_embeds,
                context_hidden_states=hidden_states,
                context_attention_mask=attention_mask,
                **kwargs,
            )[0].to(device=self.icformer.device)
        else:
            soft_prompt = []
            chunk_num = math.ceil(embeddings.shape[1] / self.max_chunk_len)
            chunk_size = math.ceil(embeddings.shape[1] / chunk_num)
            
            for index in range(chunk_num):
                chunk_embeds = embeddings[:,index*chunk_size:(index+1)*chunk_size]
                chunk_mask = torch.ones([1, query_embeds.shape[1], chunk_embeds.shape[1]])
                attention_mask = torch.cat([chunk_mask, query_mask], dim=-1).to(device=self.icformer.device)
                hidden_states = torch.cat([chunk_embeds, query_embeds], dim=1).to(device=self.icformer.device)
                chunk_soft_prompt = self.icformer(
                    query_embeds=query_embeds,
                    context_hidden_states=hidden_states,
                    context_attention_mask=attention_mask,
                    **kwargs,
                )[0][:,-self.digest_embeddings.shape[1]:]
                soft_prompt.append(chunk_soft_prompt.to(device=self.icformer.device))
            soft_prompt = torch.cat(soft_prompt, dim=1)
        return soft_prompt

    def get_pretrained_model(self):
        return self.icformer


