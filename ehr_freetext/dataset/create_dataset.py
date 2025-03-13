import json
import random
import sys
from collections import defaultdict
import tqdm

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def sample_data(data):
    # 根据 task_id 分组，对于任务 1,2,4,5 按 output 分组，
    # 其他按 task_id 分组。
    group_samples = []
    tasks_split = {"1", "2", "4", "5"}
    # 暂存每个 task_id 或 (task_id, output) 分组
    groups = defaultdict(list)
    for item in data:
        task_id = item.get("task_id")
        if task_id in tasks_split:
            output_val = item.get("output", "")
            key = (task_id, output_val)
        else:
            key = task_id
        groups[key].append(item)
    
    # 对每个分组做采样
    for key, items in groups.items():
        if isinstance(key, tuple):  # task_id in {"1","2","4","5"}
            target_samples = 500
        else:
            target_samples = 1000
        
        if len(items) < target_samples:
            sampled = items
        else:
            sampled = random.sample(items, target_samples)
        # 划分训练集和测试集，测试集:训练集 = 1:9
        test_size = max(1, len(sampled) // 10)  # 至少一个数据
        random.shuffle(sampled)
        test_items = sampled[:test_size]
        train_items = sampled[test_size:]
        group_samples.append(("train", train_items))
        group_samples.append(("test", test_items))
    return group_samples

def write_jsonl(filename, items):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in items:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")

def main():
    if len(sys.argv) < 2:
        print("使用方法: python sample_data.py <input_jsonl_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    data = load_jsonl(input_file)
    grouped_samples = sample_data(data)
    
    train_data = []
    test_data = []
    for tag, items in tqdm.tqdm(grouped_samples):
        if tag == "train":
            train_data.extend(items)
        else:
            test_data.extend(items)
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    write_jsonl("train_16000.jsonl", train_data)
    write_jsonl("test_16000.jsonl", test_data)
    print(f"采样完成，训练集样本数: {len(train_data)}，测试集样本数: {len(test_data)}")

if __name__ == "__main__":
    main()