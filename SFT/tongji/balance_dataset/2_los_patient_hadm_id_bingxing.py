import os
import json
from collections import defaultdict
import pandas as pd
import tqdm
import multiprocessing as mp
from functools import partial
import datetime

def process_batch(batch_data, root_dir):
    batch_stats = defaultdict(int)
    batch_yes_records = []
    batch_no_records = []
    
    for _, row in batch_data.iterrows():
        task_id = str(int(row['task']))
        patient_id = int(row['subject_id'])
        hadm_id = int(row['hadm_id'])
        
        if task_id != '2':
            continue
            
        patient_dir = os.path.join(root_dir, str(patient_id))
        if not os.path.isdir(patient_dir):
            continue
            
        hadms_dir = os.path.join(patient_dir, "hadms")
        if not os.path.isdir(hadms_dir):
            continue
        
        # 处理hadms目录下的jsonl文件
        for jsonl_file in os.listdir(hadms_dir):
            if jsonl_file.endswith(".jsonl") and jsonl_file.startswith(str(hadm_id)):
                file_path = os.path.join(hadms_dir, jsonl_file)
                with open(file_path, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            file_name = data['file_name']
                            if file_name == 'admissions':
                                admittime_str = data["items"][0]["admittime"]
                                dischtime_str = data["items"][0]["dischtime"]
                                admittime = datetime.datetime.strptime(admittime_str, "%Y-%m-%d %H:%M:%S")
                                dischtime = datetime.datetime.strptime(dischtime_str, "%Y-%m-%d %H:%M:%S")
                                hosp_day = (dischtime - admittime).days
                                record = {
                                            "patient_id": patient_id,
                                            "hadm_id": hadm_id,
                                            "task": task_id
                                        }
                                if hosp_day > 7:
                                    flag = 1
                                    batch_stats[flag] += 1
                                    batch_yes_records.append(record)
                                else:
                                    flag = 0
                                    batch_stats[flag] += 1
                                    batch_no_records.append(record)
                        except json.JSONDecodeError:
                            continue
                            
    return batch_stats, batch_yes_records, batch_no_records

def count_expire_flags(root_dir):
    # 读取数据
    patient_ids = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/bohao_mimic_project/supplementary_files/new_split/train.csv"
    df = pd.read_csv(patient_ids)
    
    # 分批处理
    num_cores = mp.cpu_count()
    batch_size = len(df) // num_cores
    batches = [df[i:i + batch_size] for i in range(0, len(df), batch_size)]
    
    # 创建进程池
    pool = mp.Pool(processes=num_cores)
    process_func = partial(process_batch, root_dir=root_dir)
    
    # 并行处理
    results = list(tqdm.tqdm(pool.imap(process_func, batches), total=len(batches), desc="处理进度"))
    pool.close()
    pool.join()
    
    # 合并结果
    total_stats = defaultdict(int)
    yes_records = []
    no_records = []
    
    for batch_stats, batch_yes, batch_no in results:
        for k, v in batch_stats.items():
            total_stats[k] += v
        yes_records.extend(batch_yes)
        no_records.extend(batch_no)
    
    # 输出结果
    print("\n总体统计结果:")
    print(f"no 的总数: {total_stats[0]}")
    print(f"yes 的总数: {total_stats[1]}")
    
    # 保存结果
    for filename, records in [("2_yes.jsonl", yes_records),
                            ("2_no.jsonl", no_records)]:
        with open(filename, 'w') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
        print(f"\n记录已保存到 {filename}")
    
    return total_stats

if __name__ == "__main__":
    root_dir = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/my_MIMIC_IV/patients_sorted/"
    count_expire_flags(root_dir)