import pandas as pd

# 读取CSV文件
df = pd.read_csv('/Users/tangbohao/Files/科研资料/EHR-decision/mimic-code/SFT/tongji/balance_dataset/new_split/test.csv')

# 打乱数据
df_shuffled = df.sample(frac=1).reset_index(drop=True)

# 保存回原文件
df_shuffled.to_csv('/Users/tangbohao/Files/科研资料/EHR-decision/mimic-code/SFT/tongji/balance_dataset/new_split/test_shuffle.csv', index=False)

print("文件已成功打乱!")