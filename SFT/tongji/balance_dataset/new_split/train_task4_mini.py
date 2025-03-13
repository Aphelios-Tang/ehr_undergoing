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
    base_path = Path('/Users/tangbohao/Files/科研资料/EHR-decision/mimic-code/SFT/tongji/balance_dataset/')
    
   
    val_df = pd.read_csv(base_path / 'new_split/train.csv')

    

    # 处理val数据
    val_selected = []

    task_data = val_df[val_df['task'] == 5][['subject_id', 'hadm_id', 'task']]

    mini_data = task_data.sample(n=6000, random_state=42)


    # 保存结果
    output_path = base_path / 'new_split/train_task5_mini.csv'
    mini_data.to_csv(output_path, index=False)


if __name__ == "__main__":
    process_data()