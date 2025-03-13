import os
import gc
import math
import json
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
# from peft import PeftModel
from utils import current_date_time
from icformer import ICFormerModel

from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re
import datetime

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
        number_encoder_checkpoint=None, # 新增参数：数字编码器 checkpoint 路径
        number_decoder_checkpoint=None, # 新增参数：数字解码器 checkpoint 路径
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
                if (epoch+1) % self.eval_interval == 0:
                    metrics = self.evaluate(self.eval_samples)
                    self.eval_metrics.append({
                        'epoch': epoch,
                        'step': step+1,
                        'metrics': {k: v for k, v in metrics.items() if k != 'examples'}  # 不保存所有样本到评估记录
                    })
                    self.log_evaluation_metrics(metrics)
                    tqdm.write(f"Evaluation at step {step+1}: BLEU-1={metrics['bleu1']:.4f}, Number accuracy={metrics['number_accuracy']:.4f}")

                if (epoch+1) % self.save_interval == 0:
                    self.save(epoch, step+1)

                start_step = 0

    def evaluate(self, num_samples=5):
        """评估模型重建质量，计算BLEU分数和数字重建准确率"""
        self.model.eval()
        
        bleu1_scores = []
        bleu2_scores = []
        bleu4_scores = []
        number_accuracies = []
        
        all_examples = []

        indices = random.sample(range(len(self.dataset)), min(num_samples, len(self.dataset)))
        
        for idx in indices:
            original_context = self.dataset.raw_data[idx] # 获取原始文本 (未处理占位符)
            processed_context, number_info = self.dataset[idx] # 获取预处理后的文本和数字信息
            
            # 生成重建文本
            with torch.no_grad():
                prediction, decoded_numbers = self.process_sample(processed_context, number_info)


            # 计算BLEU分数
            reference_tokens = word_tokenize(processed_context.lower()) # 使用原始文本
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
            
            # 计算数字重建准确率 (需要比较原始数字和重建数字)
            orig_numbers = [item['value'] for item in number_info] # 从 number_info 中提取原始数字值
            pred_numbers = [item.item() for item in decoded_numbers]

            if len(orig_numbers) > 0:
                # 计算匹配的数字比例 (这里假设解码后的数字顺序与原始数字顺序一致，实际情况可能需要更复杂的匹配)
                matched = 0
                min_len = min(len(orig_numbers), len(pred_numbers))
                for i in range(min_len):
                    if orig_numbers[i] == pred_numbers[i]: # 简单比较字符串值
                        matched += 1
                accuracy = matched / len(orig_numbers) if len(orig_numbers) > 0 else 0
                number_accuracies.append(accuracy)
            
            # 保存原文和重建文本（限制字符数以避免日志过大）
            all_examples.append({
                "original": original_context, # 保存原始文本
                "reconstruction": prediction,
                "bleu1": bleu1,
                "reconstruction_num": pred_numbers,
                "number_accuracy": accuracy if len(orig_numbers) > 0 else "No number"  # 如果没有数字则为1.0
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
        """从文本中提取所有数字 (这里修改为提取占位符)"""
        # pattern = r'\b\d+(?:\.\d+)?\b' # 原始的数字 pattern
        pattern = r"<|image_pad|>" # 匹配占位符
        return re.findall(pattern, text)
    
    def process_sample(self, processed_context, number_info):
        """处理单个样本并返回重建文本 (需要修改以处理数字)"""
        context_ids = self.model.tokenizer(processed_context)['input_ids']

        if len(context_ids) > self.model.max_seq_len:
            last_start = len(context_ids) - self.model.max_seq_len
            start = random.randint(0, last_start)
            context_ids = context_ids[start:start+self.model.max_seq_len]

        context_embeds = self.model.convert_ids_to_embeds(context_ids) # 文本 token embeddings
        number_embeddings = self.model.encode_numbers(number_info) # 数字 embeddings
        place_holder_index = []
        for i, token_id in enumerate(context_ids):
            if token_id == self.model.number_placeholder_id:
                place_holder_index.append(i)

        if(len(number_embeddings) > 0):
            for i, index in enumerate(place_holder_index):
                context_embeds[0][index] = number_embeddings[i]

        soft_prompt = self.model.get_soft_prompt(inputs_embeds=context_embeds, use_chunk=self.model.use_chunk) # 使用原始文本 embedding

        restruction_text, decoder_number = self.model.generate_num(soft_prompt, self.model.AE, place_holder_index, max_new_tokens=512)

        output = self.replace_numbers_in_text(restruction_text, decoder_number) # 替换数字

        return output, decoder_number

    def replace_numbers_in_text(self, text, numbers_tensor_list):
 
        replaced_text = text
        numbers_used = []
        number_index = 0

        while "<|image_pad|>" in replaced_text:
            if number_index < len(numbers_tensor_list):
                current_tensor = numbers_tensor_list[number_index]

                try:
                    # Extract the numerical value from the tensor and convert to float
                    replacement_number_float = current_tensor.item()
                    replacement_number_str = "{:.1f}".format(replacement_number_float) # Convert float to string for replacement

                    replaced_text = replaced_text.replace("<|image_pad|>", replacement_number_str, 1) # Replace only the first occurrence
                    numbers_used.append(replacement_number_float) # Store the float value
                    number_index += 1

                except Exception as e:
                    print(f"Error processing tensor at index {number_index}: {e}")
                    print(f"Tensor causing error: {current_tensor}")
                    break 
            else:
 
                break

        return replaced_text
    
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
        if hasattr(self.model, 'number_encoder'): # 保存数字编码器
            param['number_encoder_state_dict'] = self.model.number_encoder.state_dict()
        if hasattr(self.model, 'number_decoder'): # 保存数字解码器
            param['number_decoder_state_dict'] = self.model.number_decoder.state_dict()
        
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

# finish
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

class NumberEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim # 512
        self.output_dim = output_dim # 3584
        self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        self._initialize_parameters()
    def _initialize_parameters(self):
        for m in self.modules(): # 遍历所有子模块 (包括 self 本身)
            if isinstance(m, nn.Linear): # 判断是否是线性层
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu') # Kaiming Uniform 初始化权重 (ReLU 激活函数常用)
                if m.bias is not None: # 偏置可能为 None (如果 bias=False)
                    torch.nn.init.zeros_(m.bias) # 偏置初始化为 0

    def forward(self, number_value, devices):
        number = float(number_value) / 100.0 # 数值
        value_tensor = torch.tensor([number], dtype=torch.bfloat16, device=devices)
        return self.encoder(value_tensor)  # torch.Size([3584])
 
class NumberDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.decoder_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
            )
        self._initialize_parameters()
    def _initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
    def forward(self, number_embedding):
        decoder_output = self.decoder_mlp(number_embedding) # 输出 shape 
        return decoder_output.squeeze(0).squeeze(0) * 100.0 

class MIMICDataset(Dataset):
    def __init__(self, file, tokenizer): # 构造函数接受 tokenizer
        self.raw_data = self.parse_file(file)
        self.number_placeholder = "<|image_pad|>" # 定义数字占位符，与 tokenizer 中添加的保持一致
        self.tokenizer = tokenizer # 保存 tokenizer 实例
        self.number_placeholder_id = tokenizer.convert_tokens_to_ids([self.number_placeholder])[0] # 获取占位符 token id

    def parse_file(self, file):
        ret = []
        record = set()
        with open(file, 'r') as f:
            for line in f:
                data = json.loads(line)['text']
                # data随机取1024长度的子串
                if len(data) > 1024:
                    start = random.randint(0, len(data) - 1024)
                    data = data[start:start+1024]
                if data not in record:
                    record.add(data)
                    ret.append(data)
        return ret

    def __getitem__(self, index):
        data = self.raw_data[index]
        processed_text, number_info = self.preprocess_text_with_numbers(data) # 新增预处理函数
        tokenized_output = self.tokenizer(processed_text, return_offsets_mapping=True) # 获取 offset mapping
        input_ids = tokenized_output['input_ids']
        offset_mapping = tokenized_output['offset_mapping']

        # 更新 number_info 中的 token 级别的位置信息
        for number_item in number_info:
            char_start = number_item['start']
            char_end = number_item['end']
            token_start = -1
            token_end = -1
            for token_index, (token_char_start, token_char_end) in enumerate(offset_mapping):
                if token_char_start <= char_start and token_char_end >= char_start and token_start == -1:
                    token_start = token_index
                if token_char_start <= char_end and token_char_end >= char_end: # 修改结束位置判断条件
                    token_end = token_index + 1 # token end index 是 exclusive 的
                    break # 找到结束 token 后就可以停止了
            number_item['token_start'] = token_start
            number_item['token_end'] = token_end

        return processed_text, number_info # 返回处理后的文本和数字信息

    def __len__(self):
        return len(self.raw_data)

    def shuffle(self):
        random.shuffle(self.raw_data)

    def preprocess_text_with_numbers(self, text):
        number_info = []
        processed_text = text
        number_pattern = re.compile(
        r'(\b\d+\.\d+\b|'  # 小数
        r'(?<![-\d:])\b\d+\b(?![:\d-]))'  # 整数，使用负向预查和回顾排除日期时间中的数字
    )

        for match in reversed(list(number_pattern.finditer(text))): # 使用 reversed 避免索引错乱
            start, end = match.span()
            number_value = match.group(0)
            number_type = self.determine_number_type(number_value) # 仍然判断类型，但只对 decimal 和 integer 进行替换

            if number_type in ['decimal', 'integer']: # 只替换 decimal 和 integer 类型
                number_info.append({
                    'value': number_value,
                    'start': start, # 字符起始位置
                    'end': end,   # 字符结束位置
                    'type': number_type
                })
                processed_text = processed_text[:start] + self.number_placeholder + processed_text[end:] # 替换为占位符

        number_info.reverse() # 恢复原始顺序
        return processed_text, number_info


    def determine_number_type(self, number_value):
        """
        更精确地判断数字类型，优先匹配日期和时间格式，避免混淆。
        """
        if re.match(r'\b\d{4}-\d{2}-\d{2}\b', number_value): # 精确匹配日期格式 yyyy-mm-dd，添加 word boundary \b
            return 'date'
        elif re.match(r'\b(?:[01]\d|2[0-3]):(?:[0-5]\d):(?:[0-5]\d)\b', number_value): # 精确匹配时间格式 hh:mm:ss，添加 word boundary \b
            return 'time'
        elif re.match(r'\b\d+\.\d+\b', number_value): # 精确匹配小数，添加 word boundary \b
            return 'decimal'
        elif re.match(r'\b\d+\b', number_value): # 精确匹配整数，添加 word boundary \b
            return 'integer'

class ICFormer(BaseModel):
    def __init__(
        self, 
        icformer:ICFormerModel, 
        language_model:PreTrainedModel, 
        tokenizer:PreTrainedTokenizer,
        dataset: MIMICDataset,
        number_encoder_checkpoint=None, # 新增参数：数字编码器 checkpoint 路径
        number_decoder_checkpoint=None, # 新增参数：数字解码器 checkpoint 路径
    ):
        super().__init__(language_model=language_model, tokenizer=tokenizer)
        self.icformer = icformer
        self.dataset = dataset # 保存 dataset 实例

        self.digest_embeddings = nn.Parameter(torch.zeros([1, icformer.config.num_query_tokens, icformer.config.hidden_size], device=icformer.device, dtype=icformer.dtype))
        self.AE = nn.Parameter(torch.randn([1, 32, icformer.config.hidden_size], device=language_model.device, dtype=language_model.dtype))
        self.FT = nn.Parameter(torch.randn([1, 32, icformer.config.hidden_size], device=language_model.device, dtype=language_model.dtype))

        self.max_seq_len = 512
        self.max_chunk_len = 512
        self.use_chunk = True
        self.encode = False

        self.dtype = language_model.dtype
        self.device = language_model.device

        # 数字编码器和解码器
        self.number_encoder = NumberEncoder(
            input_dim=1, 
            hidden_dim=512,
            output_dim=language_model.config.hidden_size
        ).to(language_model.device, language_model.dtype) 
        self.number_decoder = NumberDecoder(
            input_dim=152064,
            hidden_dim=512,
            output_dim=1 
        ).to(language_model.device, language_model.dtype) 

        # 获取占位符 token id 
        self.number_placeholder_id = tokenizer.convert_tokens_to_ids(['<|image_pad|>'])[0]

        # 加载数字编码器和解码器 checkpoint 并冻结参数 (在 ICFormer 初始化时加载)
        if number_encoder_checkpoint and os.path.exists(number_encoder_checkpoint):
            encoder_state_dict = torch.load(os.path.join(number_encoder_checkpoint, "number_encoder.pt"), map_location=language_model.device)
            self.number_encoder.load_state_dict(encoder_state_dict)
            tqdm.write(f"Loaded NumberEncoder checkpoint from {number_encoder_checkpoint}")
        else:
            tqdm.write("NumberEncoder checkpoint not provided or not found, training from scratch.")

        if number_decoder_checkpoint and os.path.exists(number_decoder_checkpoint):
            decoder_state_dict = torch.load(os.path.join(number_decoder_checkpoint, "number_decoder.pt"), map_location=language_model.device)
            self.number_decoder.load_state_dict(decoder_state_dict)
            tqdm.write(f"Loaded NumberDecoder checkpoint from {number_decoder_checkpoint}")
        else:
            tqdm.write("NumberDecoder checkpoint not provided or not found, training from scratch.")
    
    def generate_num(self, soft_prompt, AE, place_holder_index, max_new_tokens=512, **kwargs):
        
        inputs_embeds = torch.cat([soft_prompt, AE], dim=1)
        inputs_embeds = inputs_embeds.to(dtype=self.language_model.dtype)
        past_key_values = None
        output_ids = []
        decode_number = []
        # 数字解码
        # for i in number_embeddings:
        #     number_embedding = i
        #     value = self.number_decoder(number_embedding) # 解码数字
        #     decode_number.append(value)


        for i in range(max_new_tokens):
            output = self.language_model(inputs_embeds=inputs_embeds, past_key_values=past_key_values, use_cache=True)

            next_token_id = output.logits[0][-1].argmax(dim=-1) 
            output_ids.append(next_token_id)

            if i in place_holder_index:
                number_embedding = output.logits[0][-1].to(torch.bfloat16)
                value = self.number_decoder(number_embedding) # 解码数字
                decode_number.append(value)

            if next_token_id == self.tokenizer.eos_token_id:
                break
            past_key_values = output.past_key_values
            next_embeds = self.language_model.get_input_embeddings()(torch.tensor([[next_token_id]], device=self.language_model.device))
            inputs_embeds = next_embeds
        restruction_str = self.tokenizer.decode(output_ids, skip_special_tokens=True, **kwargs)
        
        return restruction_str, decode_number


    def train_step(self, step, data):
        context, number_info = data 

        context_ids = self.tokenizer(context)['input_ids']
        place_holder_index = []
        for i, token_id in enumerate(context_ids):
            if token_id == self.number_placeholder_id:
                place_holder_index.append(i)
        if len(context_ids) > self.max_seq_len: # random split
            last_start = len(context_ids) - self.max_seq_len
            start = random.randint(0, last_start)
            context_ids = context_ids[start:start+self.max_seq_len]


        label_ids = context_ids + [self.tokenizer.eos_token_id]
        context_embeds = self.convert_ids_to_embeds(context_ids) 

        number_embeddings = self.encode_numbers(number_info)
        label_embeds = self.convert_ids_to_embeds(label_ids)

        if len(number_embeddings) > 0:
            for i, index in enumerate(place_holder_index):
                context_embeds[0][index] = number_embeddings[i] # 融入了number_embeddings

        soft_prompt = self.get_soft_prompt(inputs_embeds=context_embeds, use_chunk=self.use_chunk) # 使用原始文本 embedding'
        
        inputs_embeds = torch.cat([soft_prompt, self.AE, label_embeds], dim=1)
        # shifted right
        logits = self.language_model(inputs_embeds=inputs_embeds).logits[0][-len(label_ids)-1:-1]
        # print(logits[1].shape) # torch.Size([152064])词表每个词的概率

        ae_loss = F.cross_entropy(logits, torch.tensor(label_ids, device=logits.device), reduction="none")

        num_logits = logits[place_holder_index] # [9, 152064] # 找到所有占位符应该的token位置

        number_loss = 0.
        for i in range(num_logits.shape[0]):
            number_embedding = num_logits[i].to(torch.bfloat16) # torch.Size([152064]) 
            number_item = number_info[i]
            decoded_values = self.number_decoder(number_embedding) # 解码数字
            original_value = number_item['value']
            try:
                original_value_float = float(original_value)
                number_loss += F.mse_loss(decoded_values.unsqueeze(0) / 100., torch.tensor([original_value_float]).float().to(decoded_values.device) / 100.) # MSE Loss for numerical values
            except ValueError:
                pass
        number_loss /= len(number_info) if len(number_info) > 0 else 1

        number_loss = torch.tensor([number_loss], device=logits.device, dtype=torch.bfloat16) # 转换为 tensor
        total_loss = ae_loss + number_loss / 100.0

        print("number_loss:", number_loss)

        return ae_loss
    
    def encode_numbers(self, number_info):
        """
        编码数字信息为 embeddings.
        """
        number_embeddings = []
        for number_item in number_info:
            value = number_item['value']
            embedding = self.number_encoder(value, self.device) # 使用数字编码器, 传递位置索引
            number_embeddings.append(embedding)
        return number_embeddings # 返回数字 embedding 列表
    
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
        pretrained_model = self.icformer
        pretrained_model.number_encoder_state_dict = self.number_encoder.state_dict()
        pretrained_model.number_decoder_state_dict = self.number_decoder.state_dict()
        return pretrained_model


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

