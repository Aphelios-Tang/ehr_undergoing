import pandas as pd

# 读取CSV文件
val_df = pd.read_csv('train.csv')

# 遍历每一行
print("\n逐行显示数据:")
for index, row in val_df.iterrows():
    subject_id = int(row['subject_id'])
    hadm_id = int(row['hadm_id'])
    task = str(int(row['task']))
    print(subject_id, hadm_id, task)
    print("-" * 50)  # 分隔线

    # 可选：每10行暂停一次，按回车继续
    if (index + 1) % 10 == 0:
        input("按回车键继续...")