import pandas as pd
import json
import random
from pathlib import Path

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return pd.DataFrame(data)

def process_data():
    base_path = Path('/Users/tangbohao/Files/科研资料/EHR-decision/mimic-code/SFT/tongji/balance_dataset')
    
    # 读取所有数据文件
    death_df = read_jsonl(base_path / '1_death_records.jsonl')
    not_death_df = read_jsonl(base_path / '1_not_death_records.jsonl')
    yes_df = read_jsonl(base_path / '2_yes.jsonl')
    no_df = read_jsonl(base_path / '2_no.jsonl')
    val_df = pd.read_csv(base_path / 'new_split/train.csv')

    # 随机采样
    sample_size = 3000
    dfs = [death_df, not_death_df, yes_df, no_df]
    sampled_dfs = []
    
    for df in dfs:
        if len(df) > sample_size:
            sampled_df = df.sample(n=sample_size, random_state=42)
        else:
            sampled_df = df
        # 重命名patient_id为subject_id
        if 'patient_id' in sampled_df.columns:
            sampled_df = sampled_df.rename(columns={'patient_id': 'subject_id'})
        sampled_dfs.append(sampled_df)

    # 处理val数据
    val_selected = []
    for task in range(3, 10):
        task_data = val_df[val_df['task'] == task][['subject_id', 'hadm_id', 'task']]
        if len(task_data) >= 6000:
            val_selected.append(task_data.sample(n=6000, random_state=42))
        else:
            val_selected.append(task_data)

    # 合并所有数据
    final_df = pd.concat([*sampled_dfs, *val_selected], ignore_index=True)
     # 随机打乱数据
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 保存结果
    output_path = base_path / 'new_split/train_mini.csv'
    final_df.to_csv(output_path, index=False)
    
    print(f"Total records: {len(final_df)}")
    for task in final_df['task'].unique():
        print(f"Task {task}: {len(final_df[final_df['task'] == task])} records")

if __name__ == "__main__":
    process_data()