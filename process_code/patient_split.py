import os
import random
import pandas as pd

# 根目录，患者数据所在路径
ROOT_DIR = "/mnt/petrelfs/wuchaoyi/my_MIMIC_IV_2.2/patients_sorted"

# 获取所有患者的名字
patients = os.listdir(ROOT_DIR)

# 打乱患者顺序
random.shuffle(patients)

# 划分比例：80% 训练集, 10% 验证集, 10% 测试集
train_size = int(0.8 * len(patients))
val_size = int(0.1 * len(patients))
test_size = len(patients) - train_size - val_size

# 划分数据
train_patients = patients[:train_size]
val_patients = patients[train_size:train_size + val_size]
test_patients = patients[train_size + val_size:]

# 将患者名单保存到CSV文件中
train_df = pd.DataFrame(train_patients, columns=["Patient_ID"])
val_df = pd.DataFrame(val_patients, columns=["Patient_ID"])
test_df = pd.DataFrame(test_patients, columns=["Patient_ID"])

# 保存为CSV文件
train_df.to_csv("train_patients.csv", index=False)
val_df.to_csv("val_patients.csv", index=False)
test_df.to_csv("test_patients.csv", index=False)

print("Patients split and saved into train_patients.csv, val_patients.csv, and test_patients.csv")
