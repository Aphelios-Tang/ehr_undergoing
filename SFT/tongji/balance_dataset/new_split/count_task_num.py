import pandas as pd

# 读取CSV文件
df = pd.read_csv('/Users/tangbohao/Files/科研资料/EHR-decision/mimic-code/SFT/tongji/balance_dataset/new_split/train.csv')

# 统计每个task的数量
task_counts = df['task'].value_counts().sort_index()

# 计算总数
total_count = len(df)

# 打印结果
print("每个task的数量:")
for task, count in task_counts.items():
    print(f"Task {task}: {count}")
    
print("\n总样本数:", total_count)

# 可以选择绘制柱状图来可视化
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
task_counts.plot(kind='bar')
plt.title('Task Distribution')
plt.xlabel('Task')
plt.ylabel('Count')
plt.tight_layout()
plt.show()