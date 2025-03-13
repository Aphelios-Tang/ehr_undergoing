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
from utils import current_date_time
from icformer import ICFormerModel

from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re

nltk_data_path = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/longcontext/model/nltk"
nltk.data.path.append(nltk_data_path)

class Trainer:
    def __init__(
        self,
        model, 
        dataset, 
        optimizer,
        scheduler=None,
        max_epoch=1,
        save_interval=5000,
        save_dir='./output',
        save_optimizer=False,
        save_scheduler=False,
        avg_level='token',
        gradient_accumulation=1,
        shuffle=True,
        clip_grad=True,
        max_norm=2,
        eval_interval=5,
        eval_samples=5,
    ):
        """
        A pytorch-lightning style trainer for training models.
        'save_optimizer' is a boolean value to save the optimizer state dict.
        'avg_level' is a string value to indicate how to average loss during gradient accumulation. You can choose 'token' or 'sentence'.
        """
        self.max_epoch = max_epoch
        self.save_interval = save_interval
        self.save_dir = save_dir
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.model = model
        self.dataset = dataset
        self.steps_per_epoch = len(dataset)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_accumulation = gradient_accumulation
        self.max_norm = max_norm
        self.shuffle = shuffle
        self.clip_grad = clip_grad

        self.eval_interval = eval_interval
        self.eval_samples = eval_samples

        self.handler = GradHandler(avg_level=avg_level)
        self.log_record = []
        self.log_file = os.path.join(self.save_dir, f"{current_date_time()}.log")
        self.eval_metrics = []  # 记录评估指标

        os.makedirs(self.save_dir, exist_ok=True)

    def train(
        self,
        start_step=0,
        start_epoch=0,
    ):
        tqdm.write(f"Start training at {current_date_time()}.")
        for epoch in range(start_epoch, self.max_epoch):
            self.model.zero_grad()
            torch.cuda.empty_cache()
            
            if self.shuffle:
                self.dataset.shuffle()

            with tqdm(total=self.steps_per_epoch-start_step, ncols=100, unit='B') as pbar:
                for step in range(start_step, self.steps_per_epoch):
                    data = self.dataset[step]
                    loss = self.model.train_step(step, data)
                    self.handler.append(loss)
                    self.handler.backward(loss)
                    pbar.update(1)

                    if (step+1) % self.gradient_accumulation == 0 or (step+1) == self.steps_per_epoch:
                        self.handler.apply_grad(self.optimizer)
                        self.log(
                            epoch=epoch, 
                            step=step+1, 
                            loss=self.handler.compute_loss(), 
                            grad=self.handler.compute_grad_norm(self.optimizer),
                        )
                        self.optimize()
                        self.handler.clear()

                        if self.scheduler is not None:
                            self.scheduler.step(step)
                    # 定期评估模型
                    if (step+1) % self.eval_interval == 0:
                        metrics = self.evaluate(self.eval_samples)
                        self.eval_metrics.append({
                            'epoch': epoch,
                            'step': step+1,
                            'metrics': {k: v for k, v in metrics.items() if k != 'examples'}  # 不保存所有样本到评估记录
                        })
                        self.log_evaluation_metrics(metrics)
                        tqdm.write(f"Evaluation at step {step+1}: BLEU-1={metrics['bleu1']:.4f}, Number accuracy={metrics['number_accuracy']:.4f}")

                    if (step+1) % self.save_interval == 0:
                        self.save(epoch, step+1)

                # if (epoch+1) % self.save_interval == 0:
                #     self.save(epoch, step+1)
                start_step = 0

    def evaluate(self, num_samples=5):
        """评估模型重建质量，计算BLEU分数和数字重建准确率"""
        self.model.eval()
        
        bleu1_scores = []
        bleu2_scores = []
        bleu4_scores = []
        number_accuracies = []
        
        all_examples = []

        # 随机选择样本进行评估
        indices = random.sample(range(len(self.dataset)), min(num_samples, len(self.dataset)))
        
        for idx in indices:
            context = self.dataset[idx]
            
            # 生成重建文本
            with torch.no_grad():
                prediction = self.process_sample(context)
            
            # 计算BLEU分数
            reference_tokens = word_tokenize(context.lower())
            hypothesis_tokens = word_tokenize(prediction.lower())
            
            # 使用平滑函数避免零精度问题
            smoothie = SmoothingFunction().method1
            
            # 计算BLEU分数
            reference_list = [reference_tokens]
            bleu1 = sentence_bleu(reference_list, hypothesis_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
            bleu2 = sentence_bleu(reference_list, hypothesis_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
            bleu4 = sentence_bleu(reference_list, hypothesis_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
            
            bleu1_scores.append(bleu1)
            bleu2_scores.append(bleu2)
            bleu4_scores.append(bleu4)
            
            # 计算数字重建准确率
            orig_numbers = self.extract_numbers(context)
            pred_numbers = self.extract_numbers(prediction)
            
            if len(orig_numbers) > 0:
                # 计算匹配的数字比例
                matched = sum(1 for n in pred_numbers if n in orig_numbers)
                accuracy = matched / len(orig_numbers) if len(orig_numbers) > 0 else 0
                number_accuracies.append(accuracy)
            
            # 保存原文和重建文本（限制字符数以避免日志过大）
            all_examples.append({
                "original": context,
                "reconstruction": prediction,
                "bleu1": bleu1,
                "number_accuracy": accuracy if len(orig_numbers) > 0 else 1.0  # 如果没有数字则为1.0
            })
            
        # 计算平均分数
        avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0
        avg_bleu2 = sum(bleu2_scores) / len(bleu2_scores) if bleu2_scores else 0
        avg_bleu4 = sum(bleu4_scores) / len(bleu4_scores) if bleu4_scores else 0
        avg_number_accuracy = sum(number_accuracies) / len(number_accuracies) if number_accuracies else 0
        
        self.model.train()
        
        return {
            'bleu1': avg_bleu1,
            'bleu2': avg_bleu2,
            'bleu4': avg_bleu4,
            'number_accuracy': avg_number_accuracy,
            'examples': all_examples 
        }
    
    @staticmethod
    def extract_numbers(text):
        """从文本中提取所有数字"""
        # 匹配整数和小数
        pattern = r'\b\d+(?:\.\d+)?\b'
        return re.findall(pattern, text)
    
    def process_sample(self, context):
        """处理单个样本并返回重建文本"""
        context_ids = self.model.tokenizer(context)['input_ids']

        if len(context_ids) > self.model.max_seq_len:
            last_start = len(context_ids) - self.model.max_seq_len
            start = random.randint(0, last_start)
            context_ids = context_ids[start:start+self.model.max_seq_len]

        context_embeds = self.model.convert_ids_to_embeds(context_ids)
        
        soft_prompt = self.model.get_soft_prompt(inputs_embeds=context_embeds, use_chunk=self.model.use_chunk)
        # pre_embeds = self.model.convert_ids_to_embeds(self.model.tokenizer(self.model.tokenizer.bos_token)['input_ids'])
        
        # prompt = "Repeat the contents carefully."
        # prompt_ids = self.model.tokenizer(prompt)['input_ids']
        # prompt_embeds = self.model.convert_ids_to_embeds(prompt_ids)

        if hasattr(self.model, 'encode') and self.model.encode:
            with torch.no_grad():
                context_embeds = self.model.language_model.model(inputs_embeds=context_embeds)[0]

        inputs_embeds = torch.cat([soft_prompt, self.model.AE], dim=1)
        
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds, 
            max_new_tokens=512,
            do_sample=True,
            top_k=50
        )
        
        prediction = self.model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return prediction
    
    def log(self, epoch, step, loss, grad, **kwargs):
        record = {}
        if loss is not None:
            record["loss"] = loss
        if grad is not None:
            record["grad_norm"] = grad
        record["learning_rate"] = self.optimizer.param_groups[0]['lr']
        record["epoch"] = epoch
        record["step"] = step
        record.update(kwargs)
        tqdm.write(f"{record}")
        self.log_record.append(record)
    
    def log_evaluation_metrics(self, metrics):
        """记录评估指标到日志文件"""
        eval_log_file = os.path.join(self.save_dir, "evaluation_metrics.json")
        
        # 读取现有日志（如果存在）
        if os.path.exists(eval_log_file):
            with open(eval_log_file, 'r') as f:
                try:
                    existing_metrics = json.load(f)
                except:
                    existing_metrics = []
        else:
            existing_metrics = []
        
            # 从所有样本中随机选择2个展示
        examples = metrics.pop('examples')
        selected_examples = random.sample(examples, min(2, len(examples)))
        
        # 添加新指标
        new_entry = {
            'timestamp': current_date_time(),
            'metrics': metrics,
            'example_reconstructions': selected_examples
        }

        existing_metrics.append(new_entry)
        
        # 保存更新后的指标
        with open(eval_log_file, 'w') as f:
            json.dump(existing_metrics, f, indent=2)

    def optimize(self):
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(parameters=self.optimizer.param_groups[0]['params'], max_norm=self.max_norm, norm_type=2)
            # torch.nn.utils.clip_grad_value_(self.optimizer.param_groups[0]['params'], clip_value=0.005)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def save(self, epoch, step):
        module_to_save = self.model.get_pretrained_model()
        save_directory = os.path.join(self.save_dir, f"checkpoint-{self.steps_per_epoch*epoch+step}")
        module_to_save.save_pretrained(save_directory=save_directory)
        param = {}
        if hasattr(self.model, 'digest_embeddings'):
            param['digest_embeddings'] = self.model.digest_embeddings
        if hasattr(self.model, 'memory_embeddings'):
            param['memory_embeddings'] = self.model.memory_embeddings
        if hasattr(self.model, 'AE'):
            param['AE'] = self.model.AE
        if hasattr(self.model, 'LM'):
            param['LM'] = self.model.LM
        if hasattr(self.model, 'FT'):
            param['FT'] = self.model.FT
        
        torch.save(param, os.path.join(save_directory, "param.pt"))
        
        if self.save_optimizer:
            torch.save(self.optimizer.state_dict(), os.path.join(save_directory, "optimizer.pt"))

        if self.save_scheduler:
            torch.save(self.scheduler.state_dict(), os.path.join(save_directory, "scheduler.pt"))

        # 保存时评估模型，并记录指标
        metrics = self.evaluate(self.eval_samples)
        self.eval_metrics.append({
            'epoch': epoch,
            'step': step,
            'checkpoint': f"checkpoint-{self.steps_per_epoch*epoch+step}",
            'metrics': {k: v for k, v in metrics.items() if k != 'examples'}  # 不保存所有样本到评估记录
        })
        self.log_evaluation_metrics(metrics)
        
        # 更新trainer状态
        trainer_state = {
            "steps_per_epoch": self.steps_per_epoch, 
            "log_history": self.log_record,
            "eval_metrics": self.eval_metrics
        }
        json.dump(trainer_state, open(os.path.join(save_directory, "trainer_state.json"), "w"), indent=2)
        
        tqdm.write(f"Saved checkpoint and evaluated model: BLEU-1={metrics['bleu1']:.4f}, Number accuracy={metrics['number_accuracy']:.4f}")

class GradHandler:
    """
    Gradient handler is designed for handling the gradient accumulation, loss averaging and gradient norm calculation.
    The handler recieves the loss of every token and accumulates them to compute the average loss.
    """
    def __init__(self, avg_level='token'):
        self.loss_list = []
        self.total_len = 0
        self.avg_level = avg_level

    def append(self, loss:torch.Tensor=None):
        if loss is not None or len(loss) > 0:
            if self.avg_level == 'token':
                self.total_len += len(loss)
                self.loss_list.append(loss.sum().item())
            elif self.avg_level == 'sentence':
                self.total_len += 1
                self.loss_list.append(loss.mean().item())

    def backward(self, loss:torch.Tensor=None):
        if loss is not None or len(loss) > 0:
            if self.avg_level == 'token':
                loss.sum().backward()
            elif self.avg_level == 'sentence':
                loss.mean().backward()

    def compute_grad_norm(self, optimizer:torch.optim.Optimizer):
        grad_norm = []
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.requires_grad and param.grad is not None:
                    grad_norm.append(param.grad.detach().norm())
        all_norm = torch.tensor(grad_norm).norm().item()
        return all_norm
    
    def compute_loss(self):
        if self.total_len == 0:
            return 0
        return sum(self.loss_list) / self.total_len
    
    def apply_grad(self, optimizer:torch.optim.Optimizer):
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.requires_grad and param.grad is not None:
                    grad = param.grad / self.total_len
                    param.grad = grad
            
    def clear(self):
        self.total_len = 0
        self.loss_list.clear()
        gc.collect()


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
        self.AE = nn.Parameter(torch.randn([1, 32, icformer.config.hidden_size], device=language_model.device, dtype=language_model.dtype))
        self.FT = nn.Parameter(torch.randn([1, 32, icformer.config.hidden_size], device=language_model.device, dtype=language_model.dtype))

        self.max_seq_len = 512
        self.max_chunk_len = 512
        self.use_chunk = True
        self.encode = False

    def train_step(self, step, data):
        context = data
        context_ids = self.tokenizer(context)['input_ids']
        if len(context_ids) > self.max_seq_len: # random split
            last_start = len(context_ids) - self.max_seq_len
            start = random.randint(0, last_start)
            context_ids = context_ids[start:start+self.max_seq_len]
        label_ids = context_ids + [self.tokenizer.eos_token_id]
        context_embeds = self.convert_ids_to_embeds(context_ids) # 1445
        
        # prompt = "Output the contents word by word and make sure your output is totally same with it. Do not output any other words. \n"
        # prompt_ids = self.tokenizer(prompt)['input_ids']
        # prompt_embeds = self.convert_ids_to_embeds(prompt_ids)
        
        # eos_embeds = self.convert_ids_to_embeds(self.tokenizer(self.tokenizer.eos_token)['input_ids'])
        
        # print("context_embeds: ", context_embeds.shape) # torch.Size([1, 512, 2048])
        # Whether to use the language model to encode the context.
        if self.encode:
            with torch.no_grad():
                context_embeds = self.language_model.model(inputs_embeds=context_embeds)[0]

        label_embeds = self.convert_ids_to_embeds(label_ids)
        soft_prompt = self.get_soft_prompt(inputs_embeds=context_embeds, use_chunk=self.use_chunk)
        inputs_embeds = torch.cat([soft_prompt, self.AE, label_embeds], dim=1)
        # shifted right
        logits = self.language_model(inputs_embeds=inputs_embeds).logits[0][-len(label_ids)-1:-1]
        ae_loss = F.cross_entropy(logits, torch.tensor(label_ids, device=logits.device), reduction="none")
        return ae_loss
    
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
        embeddings = inputs_embeds.to(device=self.icformer.device)

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
            )[0].to(device=self.language_model.device)
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
                soft_prompt.append(chunk_soft_prompt.to(device=self.language_model.device))
            soft_prompt = torch.cat(soft_prompt, dim=1)
        return soft_prompt

    def get_pretrained_model(self):
        return self.icformer


class ICFormerQA(ICFormer):
    def __init__(
        self, 
        icformer:ICFormerModel, 
        language_model:PreTrainedModel, 
        tokenizer:PreTrainedTokenizer
    ):
        super().__init__(icformer=icformer, language_model=language_model, tokenizer=tokenizer)
        self.AE.requires_grad = False
        # self.LM.requires_grad = False
        self.FT = nn.Parameter(torch.zeros([1, 1, icformer.config.hidden_size], device=language_model.device, dtype=language_model.dtype))

        self.pre_embeds = self.convert_ids_to_embeds(self.tokenizer("<s>[INST] Response the Prompt based on the below text:\n\n")['input_ids'])
        self.post_embeds = self.convert_ids_to_embeds(self.tokenizer("[/INST]")['input_ids'])

        self.alpha = 1.0 
        self.max_label_len = 65535

    def train_step(self, step, data):
        entropy_loss, kl_loss = 0, 0
        context, prompt, label = data

        label_ids = self.tokenizer(label)['input_ids'][:self.max_label_len] + [self.tokenizer.eos_token_id]
        context_ids = self.tokenizer(context)['input_ids'][:self.max_seq_len]
        prompt_ids = self.tokenizer(prompt)['input_ids']
        label_len = len(label_ids)

        label_embeds = self.convert_ids_to_embeds(label_ids)
        context_embeds = self.convert_ids_to_embeds(context_ids)
        prompt_embeds = self.convert_ids_to_embeds(prompt_ids)

        if self.encode:
            with torch.no_grad():
                context_embeds = self.language_model.model(inputs_embeds=context_embeds)[0]
        
        soft_prompt = self.get_soft_prompt(inputs_embeds=context_embeds, use_chunk=self.use_chunk)
        inputs_embeds = torch.cat([self.pre_embeds, soft_prompt, self.FT, prompt_embeds, self.post_embeds, label_embeds], dim=1)
        logits = self.language_model(inputs_embeds=inputs_embeds).logits[0][-label_len-1:-1]

        if self.alpha > 0:
            entropy_loss = F.cross_entropy(logits, torch.tensor(label_ids, device=logits.device), reduction="none")
        if self.alpha < 1:
            with torch.no_grad():
                inputs_embeds = torch.cat([self.pre_embeds, context_embeds, prompt_embeds, self.post_embeds, label_embeds], dim=1)
                target_logits = self.language_model(inputs_embeds=inputs_embeds).logits[0][-label_len-1:-1]
            kl_loss = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(target_logits, dim=-1), reduction="none").sum(-1)
        loss = self.alpha * entropy_loss + (1-self.alpha) * kl_loss
        return loss
    
    @staticmethod
    def prepare_tokenizer(tokenizer:PreTrainedTokenizer):
        tokenizer.pad_token_id = 0
        # The bos token has been added in the context.
        tokenizer.add_bos_token = False

