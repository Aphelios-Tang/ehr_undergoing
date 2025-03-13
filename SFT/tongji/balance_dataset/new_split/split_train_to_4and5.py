import pandas as pd

# 读取原始数据
df = pd.read_csv('train.csv')

# 筛选 task=4 的行
task_4_df = df[df['task'] == 5][['subject_id', 'hadm_id', 'task']]

# 保存结果
task_4_df.to_csv('train_task5.csv', index=False)

# 打印统计信息
print(f"原始数据行数: {len(df)}")
print(f"提取task=5后的行数: {len(task_4_df)}")