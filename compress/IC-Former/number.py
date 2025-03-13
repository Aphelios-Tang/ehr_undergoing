import re
import json
import torch
import torch.nn as nn
import numpy as np

def count_numbers_in_text(text):
    """
    判断文本中数字的个数，并按照规则返回结果
    - 如果没有数字，返回"没有数字"
    - 如果数字个数在1-10之间，返回B
    - 如果数字个数大于10，返回C
    
    规则：
    - 带小数点的数字算作一个数字
    - 日期格式如xxxx-xx-xx算作3个数字
    - 时间格式如xx:xx:xx算作3个数字
    """
    count = 0
    
    # 创建一个处理过的文本副本，避免重复计算
    processed_text = text
    
    # 1. 处理日期格式 (xxxx-xx-xx)
    date_pattern = r'\d{4}-\d{2}-\d{2}'
    dates = []
    for match in re.finditer(date_pattern, processed_text):
        start, end = match.span()
        dates.append((processed_text[start:end], start, end))
    
    count += len(dates) * 3  # 每个日期贡献3个数字
    # print(f"日期: {[d[0] for d in dates]}")
    
    # 从文本中移除已处理的日期（从后往前替换，避免位置变化）
    for date, start, end in sorted(dates, key=lambda x: x[1], reverse=True):
        processed_text = processed_text[:start] + " " * (end - start) + processed_text[end:]
    
    # 2. 处理时间格式 (xx:xx:xx)
    time_pattern = r'\d{2}:\d{2}:\d{2}'
    times = []
    for match in re.finditer(time_pattern, processed_text):
        start, end = match.span()
        times.append((processed_text[start:end], start, end))
    
    count += len(times) * 3  # 每个时间贡献3个数字
    # print(f"时间: {[t[0] for t in times]}")
    
    # 从文本中移除已处理的时间
    for time, start, end in sorted(times, key=lambda x: x[1], reverse=True):
        processed_text = processed_text[:start] + " " * (end - start) + processed_text[end:]
    
    # 3. 处理带小数点的数字
    decimal_pattern = r'\b\d*\.\d+\b'
    decimals = []
    for match in re.finditer(decimal_pattern, processed_text):
        start, end = match.span()
        decimals.append((processed_text[start:end], start, end))
    
    count += len(decimals)  # 每个小数算作1个数字
    # print(f"小数: {[d[0] for d in decimals]}")
    
    # 从文本中移除已处理的小数
    for decimal, start, end in sorted(decimals, key=lambda x: x[1], reverse=True):
        processed_text = processed_text[:start] + " " * (end - start) + processed_text[end:]
    
    # 4. 处理剩余的整数
    integer_pattern = r'\b\d+\b'
    integers = []
    for match in re.finditer(integer_pattern, processed_text):
        start, end = match.span()
        integers.append((processed_text[start:end], start, end))
    
    count += len(integers)
    # print(f"整数: {[i[0] for i in integers]}")
    
    # 根据数字总数返回结果
    if count == 0:
        return count, "A"
    elif 1 <= count <= 16:
        return count, "B"
    elif 17 <= count <= 64:
        return count, "C"
    else:
        return count, "D"

def count_numbers_with_positions(text):
    """
    识别文本中的数字，包括它们的类型和位置信息
    返回数字总数、类别以及位置信息
    """
    count = 0
    positions = []
    text_length = len(text) if text else 1  # 防止除零错误
    
    # 创建一个处理过的文本副本，避免重复计算
    processed_text = text
    
    # 1. 处理日期格式 (xxxx-xx-xx)
    date_pattern = r'\d{4}-\d{2}-\d{2}'
    dates = []
    for match in re.finditer(date_pattern, processed_text):
        start, end = match.span()
        dates.append((processed_text[start:end], start, end))
        # 记录日期中三个数字的位置（近似值）
        date_len = end - start
        positions.append({
            'type': 'date_year', 
            'value': match.group(0)[:4],
            'pos': start / text_length,
            'rel_pos': 0.0  # 在日期内的相对位置
        })
        positions.append({
            'type': 'date_month', 
            'value': match.group(0)[5:7],
            'pos': (start + date_len/3) / text_length,
            'rel_pos': 0.33  # 在日期内的相对位置
        })
        positions.append({
            'type': 'date_day', 
            'value': match.group(0)[8:10],
            'pos': (start + date_len*2/3) / text_length,
            'rel_pos': 0.66  # 在日期内的相对位置
        })
    
    count += len(dates) * 3  # 每个日期贡献3个数字
    
    # 从文本中移除已处理的日期（从后往前替换，避免位置变化）
    for date, start, end in sorted(dates, key=lambda x: x[1], reverse=True):
        processed_text = processed_text[:start] + " " * (end - start) + processed_text[end:]
    
    # 2. 处理时间格式 (xx:xx:xx)
    time_pattern = r'\d{2}:\d{2}:\d{2}'
    times = []
    for match in re.finditer(time_pattern, processed_text):
        start, end = match.span()
        times.append((processed_text[start:end], start, end))
        # 记录时间中三个数字的位置（近似值）
        time_len = end - start
        positions.append({
            'type': 'time_hour', 
            'value': match.group(0)[:2],
            'pos': start / text_length,
            'rel_pos': 0.0  # 在时间内的相对位置
        })
        positions.append({
            'type': 'time_minute', 
            'value': match.group(0)[3:5],
            'pos': (start + time_len/3) / text_length,
            'rel_pos': 0.33  # 在时间内的相对位置
        })
        positions.append({
            'type': 'time_second', 
            'value': match.group(0)[6:8],
            'pos': (start + time_len*2/3) / text_length,
            'rel_pos': 0.66  # 在时间内的相对位置
        })
    
    count += len(times) * 3  # 每个时间贡献3个数字
    
    # 从文本中移除已处理的时间
    for time, start, end in sorted(times, key=lambda x: x[1], reverse=True):
        processed_text = processed_text[:start] + " " * (end - start) + processed_text[end:]
    
    # 3. 处理带小数点的数字
    decimal_pattern = r'\b\d*\.\d+\b'
    decimals = []
    for match in re.finditer(decimal_pattern, processed_text):
        start, end = match.span()
        value = match.group(0)
        decimals.append((value, start, end))
        positions.append({
            'type': 'decimal', 
            'value': value,
            'pos': start / text_length,
            'rel_pos': 0.0  # 小数没有内部结构，相对位置为0
        })
    
    count += len(decimals)  # 每个小数算作1个数字
    
    # 从文本中移除已处理的小数
    for decimal, start, end in sorted(decimals, key=lambda x: x[1], reverse=True):
        processed_text = processed_text[:start] + " " * (end - start) + processed_text[end:]
    
    # 4. 处理剩余的整数
    integer_pattern = r'\b\d+\b'
    integers = []
    for match in re.finditer(integer_pattern, processed_text):
        start, end = match.span()
        value = match.group(0)
        integers.append((value, start, end))
        positions.append({
            'type': 'integer', 
            'value': value,
            'pos': start / text_length,
            'rel_pos': 0.0  # 整数没有内部结构，相对位置为0
        })
    
    count += len(integers)
    
    # 根据数字总数确定类别
    if count == 0:
        category = "A"
    elif 1 <= count <= 16:
        category = "B"
    elif 17 <= count <= 64:
        category = "C"
    else:
        category = "D"
    
    # 对位置信息按文本中的位置排序
    positions.sort(key=lambda x: x['pos'])
    
    return count, category, positions

def create_number_position_embedding(text, hidden_size=768):
    """
    创建数字位置感知嵌入
    
    参数:
    - text: 输入文本
    - hidden_size: 嵌入维度
    
    返回:
    - 数字位置嵌入特征向量 (15维)
    """
    count, category, positions = count_numbers_with_positions(text)
    
    # 创建位置直方图特征
    bins = 10
    position_histogram = np.zeros(bins)
    type_counts = {'date': 0, 'time': 0, 'decimal': 0, 'integer': 0}
    
    for pos_info in positions:
        # 更新位置直方图
        bin_idx = min(int(pos_info['pos'] * bins), bins-1)
        position_histogram[bin_idx] += 1
        
        # 更新类型计数
        if 'date' in pos_info['type']:
            type_counts['date'] += 1
        elif 'time' in pos_info['type']:
            type_counts['time'] += 1
        elif pos_info['type'] == 'decimal':
            type_counts['decimal'] += 1
        elif pos_info['type'] == 'integer':
            type_counts['integer'] += 1
    
    # 标准化直方图
    if sum(position_histogram) > 0:
        position_histogram = position_histogram / sum(position_histogram)
    
    # 创建数值特征向量 (15维)
    features = np.zeros(bins + len(type_counts) + 1)
    features[:bins] = position_histogram
    features[bins] = type_counts['date'] / max(count, 1)
    features[bins+1] = type_counts['time'] / max(count, 1)
    features[bins+2] = type_counts['decimal'] / max(count, 1)
    features[bins+3] = type_counts['integer'] / max(count, 1)
    features[-1] = count / 100.0  # 归一化数字总数
    
    return features  # 15维特征向量

class NumberPositionEmbedder(nn.Module):
    """数字位置嵌入器"""
    
    def __init__(self, hidden_size=768, expand_ratio=8, device_type = 'cuda'):
        super().__init__()
        # 15维特征 = 10维位置直方图 + 4维类型比例 + 1维数字总数
        self.feature_dim = 15
        self.device_type = device_type
        self.hidden_size = hidden_size  # 添加这行，保存hidden_size作为实例变量
        self.projector = nn.Linear(self.feature_dim, hidden_size, dtype=self.device_type)
        self.activation = nn.GELU()
        self.expand_ratio = expand_ratio

        self.expand_layer = nn.Linear(hidden_size, expand_ratio * hidden_size)
        
    def forward(self, text):
        # 获取数字特征
        features = create_number_position_embedding(text, self.hidden_size)
        
        # 获取目标设备和数据类型
        target_device = self.projector.weight.device
        target_dtype = self.projector.weight.dtype  # 使用与projector权重相同的数据类型
        
        # 创建tensor时直接指定正确的数据类型
        features_tensor = torch.tensor(features, dtype=target_dtype).to(target_device)
        
        # 确保张量维度正确
        if features_tensor.dim() == 1:
            features_tensor = features_tensor.unsqueeze(0)  # 添加batch维度 [15] -> [1, 15]
        
        # 投影到隐藏空间
        embedding = self.projector(features_tensor)  # [1, 15] -> [1, hidden_size]
        embedding = self.activation(embedding)

        # 使用扩展层将[1, hidden_size]映射为[1, 8*hidden_size]
        expanded = self.expand_layer(embedding)  # [1, 8*hidden_size]
        
        # 重塑为[1, 8, hidden_size]
        embedding = expanded.view(1, self.expand_ratio, self.hidden_size)
        # embedding = embedding.unsqueeze(0)
        
        return embedding  # [1, self.expand_ratio, hidden_size]，已经在正确的设备和数据类型上

def extract_numerical_features(text):
    """
    提取文本中数字的详细特征，包括:
    - 数字总数
    - 类别(A/B/C/D)
    - 数字类型分布（日期、时间、小数、整数）
    - 位置分布
    - 数值统计信息（平均值、最大值等）
    """
    count, category, positions = count_numbers_with_positions(text)
    
    # 基本统计信息
    result = {
        "count": count,
        "category": category,
        "type_distribution": {
            "date": 0,
            "time": 0,
            "decimal": 0,
            "integer": 0
        },
        "position_histogram": np.zeros(10),
        "values": []
    }
    
    if count == 0:
        return result
    
    # 处理所有数字
    for pos_info in positions:
        # 更新类型分布
        if 'date' in pos_info['type']:
            result["type_distribution"]["date"] += 1
        elif 'time' in pos_info['type']:
            result["type_distribution"]["time"] += 1
        elif pos_info['type'] == 'decimal':
            result["type_distribution"]["decimal"] += 1
            try:
                result["values"].append(float(pos_info['value']))
            except ValueError:
                pass
        elif pos_info['type'] == 'integer':
            result["type_distribution"]["integer"] += 1
            try:
                result["values"].append(float(pos_info['value']))
            except ValueError:
                pass
        
        # 更新位置直方图
        bin_idx = min(int(pos_info['pos'] * 10), 9)
        result["position_histogram"][bin_idx] += 1
    
    # 计算值的统计信息
    if result["values"]:
        result["value_stats"] = {
            "mean": np.mean(result["values"]),
            "median": np.median(result["values"]),
            "max": np.max(result["values"]),
            "min": np.min(result["values"]),
            "std": np.std(result["values"]) if len(result["values"]) > 1 else 0
        }
    
    # 标准化位置直方图
    if sum(result["position_histogram"]) > 0:
        result["position_histogram"] = result["position_histogram"] / sum(result["position_histogram"])
    
    return result



# 执行测试
if __name__ == "__main__":
    sample_text = "Patient (ID: 12345) had glucose levels of 127.5 mg/dL on 2023-05-12 at 08:45:30"
    features = extract_numerical_features(sample_text)
    print(json.dumps(features, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x))
    
    # 测试JSONL文件中的数据
    json_dir = "/Users/tangbohao/Files/科研资料/EHR-decision/mimic-code/ehr_freetext/reconstruction_item_0.001.jsonl"
    with open(json_dir, 'r') as f:
        file_content = f.readlines()
        i = 0
        for line in file_content:
            i += 1
            if i > 5:  # 测试前5个样本即可
                break   
            data = json.loads(line)
            text = data["text"]
            count, category = count_numbers_in_text(text)  # 保留原有函数
            features = extract_numerical_features(text)    # 提取丰富特征
            print(f"样本 {i}:")
            print(f"  数字个数: {count}, 类别: {category}")
            print(f"  类型分布: {features['type_distribution']}")
            print(f"  位置直方图: {features['position_histogram']}")
            if 'value_stats' in features:
                print(f"  值统计: {features['value_stats']}")
            print("-" * 50)
        

