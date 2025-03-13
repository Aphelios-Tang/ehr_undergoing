import json
import random
import pandas as pd
from collections import defaultdict
import tqdm
import multiprocessing as mp
from pathlib import Path

def process_line(line):
    """处理单行数据"""
    return json.loads(line)

def load_data(filename):
    """并行加载数据"""
    print("正在加载数据...")
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # 使用进程池并行处理
    with mp.Pool(processes=mp.cpu_count()) as pool:
        data = list(tqdm.tqdm(
            pool.imap(process_line, lines),
            total=len(lines),
            desc="加载数据"
        ))
    return data

def save_to_csv(data, filename):
    """保存数据到CSV"""
    df = pd.DataFrame(data)
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"已保存到 {filename}")

def main():
    # 1. 并行读取数据
    data = load_data('all_sample_tasks.jsonl')

    # 2. 按task分组
    print("正在按task分组...")
    task_groups = defaultdict(list)
    for item in tqdm.tqdm(data, desc="分组数据"):
        task_groups[item['task']].append(item)

    # 3. 为每个task划分数据
    train_data = []
    val_data = []
    test_data = []

    print("正在划分数据...")
    for task, items in tqdm.tqdm(task_groups.items(), desc="划分数据"):
        random.shuffle(items)
        train_size = int(0.8 * len(items))
        val_size = int(0.1 * len(items))
        
        train_data.extend(items[:train_size])
        val_data.extend(items[train_size:train_size + val_size])
        test_data.extend(items[train_size + val_size:])

    # 4. 保存数据
    print("正在保存数据...")
    save_to_csv(train_data, 'new_split/train.csv')
    save_to_csv(val_data, 'new_split/val.csv')
    save_to_csv(test_data, 'new_split/test.csv')

    # 5. 打印统计信息
    print(f"\n数据集大小:")
    print(f"训练集: {len(train_data)}")
    print(f"验证集: {len(val_data)}")
    print(f"测试集: {len(test_data)}")

    print("\n各任务分布:")
    for task in task_groups.keys():
        train_count = sum(1 for x in train_data if x['task'] == task)
        val_count = sum(1 for x in val_data if x['task'] == task)
        test_count = sum(1 for x in test_data if x['task'] == task)
        total = train_count + val_count + test_count
        print(f"\nTask {task}:")
        print(f"训练集: {train_count} ({train_count/total:.2%})")
        print(f"验证集: {val_count} ({val_count/total:.2%})")
        print(f"测试集: {test_count} ({test_count/total:.2%})")

if __name__ == '__main__':
    main()