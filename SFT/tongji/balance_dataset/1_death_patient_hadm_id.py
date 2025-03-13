import os
import json
from collections import defaultdict
import pandas as pd
import tqdm

def count_expire_flags(root_dir):
    total_stats = defaultdict(int)
    patient_stats = {}
    death_records_1 = []
    not_death_records_1 = []

    patient_ids = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/bohao_mimic_project/supplementary_files/new_split/train.csv"
    patients_hadm_task_id = pd.read_csv(patient_ids)
    # 遍历所有病人目录
    for index, row in tqdm.tqdm(patients_hadm_task_id.iterrows()):
    # for patient_id in tqdm.tqdm(patient_ids_list):
        task_id = str(int(row['task']))
        patient_id = int(row['subject_id'])
        hadm_id = int(row['hadm_id'])

        if task_id =='1':
            patient_dir = os.path.join(root_dir, str(patient_id))
            if not os.path.isdir(patient_dir):
                continue
                
            hadms_dir = os.path.join(patient_dir, "hadms")
            if not os.path.isdir(hadms_dir):
                continue
                
            # 初始化该病人的统计数据
            patient_stats[patient_id] = defaultdict(int)
            
            # 处理hadms目录下的所有jsonl文件
            for jsonl_file_0 in os.listdir(hadms_dir):
                if jsonl_file_0.endswith(".jsonl") and jsonl_file_0.startswith(str(hadm_id)):
                    jsonl_file = jsonl_file_0    
            file_path = os.path.join(hadms_dir, jsonl_file)
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if 'items' in data:
                            for item in data['items']:
                                if 'hospital_expire_flag' in item:
                                    flag = item['hospital_expire_flag']
                                    patient_stats[patient_id][flag] += 1
                                    total_stats[flag] += 1
                                    if flag == 1:
                                        death_records_1.append({
                                            "patient_id": patient_id,
                                            "hadm_id": hadm_id, 
                                            "task": task_id
                                        })
                                    elif flag == 0:
                                        not_death_records_1.append({
                                            "patient_id": patient_id,
                                            "hadm_id": hadm_id, 
                                            "task": task_id
                                        })
                    except json.JSONDecodeError:
                        continue

    # 打印统计结果
    print("\n总体统计结果:")
    print(f"hospital_expire_flag = 0 的总数: {total_stats[0]}")
    print(f"hospital_expire_flag = 1 的总数: {total_stats[1]}")

    # 保存记录到jsonl文件
    output_file = "death_records_1.jsonl"
    with open(output_file, 'w') as f:
        for record in death_records_1:
            f.write(json.dumps(record) + '\n')
    
    print(f"\n记录已保存到 {output_file}")

    output_file2 = "not_death_records_1.jsonl"
    with open(output_file2, 'w') as f:
        for record in not_death_records_1:
            f.write(json.dumps(record) + '\n')

    print(f"\n记录已保存到 {output_file2}")
    return total_stats

if __name__ == "__main__":
    root_dir = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/my_MIMIC_IV/patients_sorted/"
    count_expire_flags(root_dir)