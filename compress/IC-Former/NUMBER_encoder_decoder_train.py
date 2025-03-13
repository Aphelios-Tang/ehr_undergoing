import os
import json
import random
import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import re

from NUMBER_modules import NumberEncoder, NumberDecoder # 假设 NumberEncoder 和 NumberDecoder 定义在 NUMBER_modules.py 中
from utils import seed_everything, current_date_time

nltk_data_path = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/longcontext/model/nltk"
import nltk
nltk.data.path.append(nltk_data_path)


class NumberDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data = self.number_data(data_dir, split)
        self.number_placeholder = "NUMBER"
    def number_data(self, data_dir, split='train'):
        data = self.load_data(data_dir, split)
        number = []
        for i in data:
            for j in i:
                number.append(j)
        return number
    def load_data(self, data_dir, split):
        """
        加载数据，并提取包含数字的样本。
        """
        file_path = os.path.join(data_dir) # 假设数据文件路径直接传入 data_dir
        raw_data = []
        with open(file_path, 'r') as f:
            for line in f:
                raw_data.append(json.loads(line)['text'])

        number_data = []
        for text in raw_data:
            processed_text, number_info = self.preprocess_text_with_numbers(text)
            if number_info: # 只保留包含数字的样本
                number_data.append(number_info)
        return number_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        number_item = self.data[index] # 每个样本可能包含多个数字

        number_value = number_item['value']
        number_type = number_item['type']

        return number_value, number_type

    def preprocess_text_with_numbers(self, text):
        """
        预处理文本，识别数字并记录数字信息 (不替换占位符，因为这里只需要数字信息)。
        """
        number_info = []
        number_pattern = re.compile(
        r'(\b\d+\.\d+\b|'  # 小数
        r'(?<![-\d:])\b\d+\b(?![:\d-]))'  # 整数，使用负向预查和回顾排除日期时间中的数字
    )
        for match in number_pattern.finditer(text):
            start, end = match.span()
            number_value = match.group(0)
            number_type = self.determine_number_type(number_value)

            number_info.append({
                'value': number_value,
                'start': start,
                'end': end,
                'type': number_type
            })
        return text, number_info # 返回原始文本和数字信息

    def determine_number_type(self, number_value):
        """判断数字类型"""
        if re.match(r'\b\d{4}-\d{2}-\d{2}\b', number_value):
            return 'date'
        elif re.match(r'\b\d{2}:\d{2}:\d{2}\b', number_value):
            return 'time'
        elif re.match(r'\b\d+\.\d+\b', number_value):
            return 'decimal'
        elif re.match(r'\b\d+\b', number_value):
            return 'integer'
        return 'unknown'


def evaluate_number_modules(encoder, decoder, dataloader, device):
    """评估数字编码器和解码器，仅返回平均损失"""
    encoder.eval()
    decoder.eval()
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for number_value, number_type_str in dataloader:
            number_value_list = list(number_value) # 转换为 list
            batch_size = len(number_value_list)
            value_tensor_list = []

            for i in range(batch_size): # 遍历批次中的每个元素
                value = number_value_list[i] # 获取单个 number_value 字符串
                value_tensor_list.append(value)

            number_embeddings = []
            for i in range(batch_size):
                number_embedding = encoder(value_tensor_list[i], device)
                number_embeddings.append(number_embedding.unsqueeze(0))
            number_embeddings_batch = torch.cat(number_embeddings, dim=0)

            # 解码
            decoded_values_list = []
            loss = 0.0
            for i in range(batch_size):
                decoded_values = decoder(number_embeddings_batch[i].unsqueeze(0))
                decoded_values_list.append(decoded_values)
                target_value = None # 初始化 target_value
                
                decoded_values = decoded_values / 100.0
                target_value = torch.tensor(float(number_value_list[i]) / 100.0, dtype=torch.float32).to(args.device)
                loss += F.mse_loss(decoded_values.float(), target_value) # 针对标量值的 MSE loss


            total_loss += loss.item()
            num_samples += batch_size

    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    return avg_loss


def log_example_predictions(encoder, decoder, dataset, device, log_file, num_examples=5):
    """
    在日志文件中记录几个验证样本的预测结果。
    """
    encoder.eval()
    decoder.eval()
    log_str_examples = "\n--- Example Predictions ---"
    val = []
    # dataset中随机取num_examples个样本
    for i in range(num_examples):
        val.append(random.choice(dataset))
    number_info_list = val

    with torch.no_grad():
        for number_item in number_info_list: # 遍历每个数字
            number_value = number_item[0]


            number_embedding = encoder(number_value, device).unsqueeze(0)
            decoded_values = decoder(number_embedding)

            decoded_number_str = f"{decoded_values.item():.1f}"
            

            log_str_examples += f"\nOriginal: {number_value}, Decoded: {decoded_number_str}"
    tqdm.write(log_str_examples)
    with open(log_file, 'a') as f:
        f.write(log_str_examples + '\n')

def train_number_modules(args):
    seed_everything(args.seed)

    # 1. 数据准备
    train_dataset = NumberDataset(args.data_path, split='train')
    val_dataset = NumberDataset(args.data_path, split='val')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. 模型定义
    encoder = NumberEncoder(
        input_dim=1, 
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim
    ).to(args.device, torch.bfloat16)
    decoder = NumberDecoder(
        input_dim=args.output_dim,
        hidden_dim=args.hidden_dim,
        output_dim=1 
    ).to(args.device, torch.bfloat16)

    # 3. 优化器
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    log_file = os.path.join(args.save_dir, f"number_module_training_{current_date_time()}.log")

    # 4. 训练循环
    for epoch in range(args.epochs):
        encoder.train()
        decoder.train()
        total_train_loss = 0.0
        num_train_steps = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", ncols=100)

        for number_value, number_type_str in progress_bar:
            optimizer.zero_grad()
            number_value_list = list(number_value)

            batch_size = len(number_value_list)
            value_tensor_list = []

            for i in range(batch_size): # 遍历批次中的每个元素
                value = number_value_list[i] # 获取单个 number_value 字符串
                value_tensor_list.append(value)

            number_embeddings = []
            for i in range(batch_size):
                number_embedding = encoder(value_tensor_list[i], args.device)
                number_embeddings.append(number_embedding.unsqueeze(0))
            number_embeddings_batch = torch.cat(number_embeddings, dim=0) # [batch_size, 1, output_dim]

            # 解码
            decoded_values_list = []
            loss = 0.0
            for i in range(batch_size):
                decoded_values = decoder(number_embeddings_batch[i].unsqueeze(0))
                decoded_values_list.append(decoded_values)
                target_value = None # 初始化 target_value
                decoded_values = decoded_values / 100.0
                target_value = torch.tensor(float(number_value_list[i]) / 100.0, dtype=torch.float32).to(args.device)
                loss += F.mse_loss(decoded_values.float(), target_value) # 针对标量值的 MSE loss

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            num_train_steps += 1
            avg_train_loss = total_train_loss / num_train_steps
            progress_bar.set_postfix({"loss": f"{avg_train_loss:.4f}"})
        # 验证
        avg_val_loss = evaluate_number_modules(encoder, decoder, val_dataloader, args.device)
        log_str = f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}"
        tqdm.write(log_str)
        with open(log_file, 'a') as f:
            f.write(log_str + '\n')

        # 输出验证样本预测结果
        if (epoch + 1) % args.log_example_interval == 0: # 每隔 log_example_interval 个 epoch 输出验证样本
            log_example_predictions(encoder, decoder, val_dataset, args.device, log_file, num_examples=5)


        # 保存 checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_dir = os.path.join(args.save_dir, f"checkpoint-epoch-{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(encoder.state_dict(), os.path.join(checkpoint_dir, "number_encoder.pt"))
            torch.save(decoder.state_dict(), os.path.join(checkpoint_dir, "number_decoder.pt"))
            tqdm.write(f"Saved checkpoint to {checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Number Encoder and Decoder")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data file (jsonl)")
    parser.add_argument("--save_dir", type=str, default="./number_module_output", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension for NumberEncoder and Decoder")
    parser.add_argument("--output_dim", type=int, default=3584, help="Output dimension for NumberEncoder (should match LM embedding dim)")
    parser.add_argument("--save_interval", type=int, default=10, help="Epoch interval to save checkpoints")
    parser.add_argument("--log_example_interval", type=int, default=5, help="Epoch interval to log example predictions") # 新增参数
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device == 'cuda':
        print("CUDA is not available, switching to CPU.")
        args.device = 'cpu'

    train_number_modules(args)
