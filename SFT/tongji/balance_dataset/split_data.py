import json
import random
import pandas as pd
from collections import defaultdict

# 1. 读取数据
data = []
with open('all_sample_tasks.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

# 2. 按task分组
task_groups = defaultdict(list)
for item in data:
    task_groups[item['task']].append(item)

# 3. 为每个task划分数据
train_data = []
val_data = []
test_data = []

for task, items in task_groups.items():
    # 打乱该task的数据
    random.shuffle(items)
    
    # 计算划分点
    train_size = int(0.8 * len(items))
    val_size = int(0.1 * len(items))
    
    # 划分数据
    train_data.extend(items[:train_size])
    val_data.extend(items[train_size:train_size + val_size])
    test_data.extend(items[train_size + val_size:])

# 4. 转换为DataFrame并保存
def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

# 保存数据
save_to_csv(train_data, 'new_split/train.csv')
save_to_csv(val_data, 'new_split/val.csv')
save_to_csv(test_data, 'new_split/test.csv')

# 打印每个集合的大小
print(f"训练集大小: {len(train_data)}")
print(f"验证集大小: {len(val_data)}")
print(f"测试集大小: {len(test_data)}")

# 验证每个task的分布
for task in task_groups.keys():
    train_count = sum(1 for x in train_data if x['task'] == task)
    val_count = sum(1 for x in val_data if x['task'] == task)
    test_count = sum(1 for x in test_data if x['task'] == task)
    total = train_count + val_count + test_count
    print(f"\nTask {task}的分布:")
    print(f"训练集: {train_count} ({train_count/total:.2%})")
    print(f"验证集: {val_count} ({val_count/total:.2%})")
    print(f"测试集: {test_count} ({test_count/total:.2%})")