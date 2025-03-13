import pandas as pd
import json
import random

# 1. 读取death_records.jsonl获取死亡病人ID
death_ids = set()
with open('/Users/tangbohao/Files/科研资料/EHR-decision/mimic-code/SFT/tongji/balance_dataset/1_death_records.jsonl', 'r') as f:
    for line in f:
        record = json.loads(line)
        death_ids.add(record['patient_id'])
death_ids = list(death_ids)

# 2. 读取test_patients.csv获取测试病人ID
test_df = pd.read_csv('/Users/tangbohao/Files/科研资料/EHR-decision/mimic-code/supplementary_files/split/train_patients.csv')
test_ids = test_df['Patient_ID'].tolist()

# 3. 从测试病人中筛选出不在死亡名单中的病人
alive_ids = list(set(test_ids) - set(death_ids))

# 4. 随机抽取与死亡人数相等的存活病人
selected_alive = random.sample(alive_ids, len(death_ids))

# 5. 创建最终的DataFrame
final_data = {
    'Patient_ID': death_ids + selected_alive,
    'Task': ["1"] * len(death_ids + selected_alive)
}
final_df = pd.DataFrame(final_data)

# 6. 保存为CSV文件
final_df.to_csv('1_mortality_task.csv', index=False)

print(f"Total death cases: {len(death_ids)}")
print(f"Total selected alive cases: {len(selected_alive)}")
print(f"Total samples: {len(final_df)}")