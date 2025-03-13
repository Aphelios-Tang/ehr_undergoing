import os
import json
import tqdm
from multiprocessing import Pool, cpu_count

def read_jsonl(jsonl_dir):
    data_list = []
    with open(jsonl_dir, 'r') as file:
        for line in file:
            data_list.append(json.loads(line))
    return data_list

def process_single_record(line):
    data = json.loads(line.strip())
    subject_id = data["subject_id"]
    hadm_id = data["hadm_id"]
    hadm_file = os.path.join(patient_root_dir, str(subject_id), "hadms", f"{hadm_id}.jsonl")
    
    try:
        patient_trajectory_list = read_jsonl(hadm_file)
        target_task = []
        
        for item in patient_trajectory_list:
            file_name = item["file_name"]
            if file_name == "admissions":
                target_task.extend([(subject_id, hadm_id, str(i)) for i in [1, 2, 4]])
            elif file_name == "diagnoses_icd":
                target_task.append((subject_id, hadm_id, "3"))
            elif file_name == "ed":
                target_task.append((subject_id, hadm_id, "5"))
            elif file_name == "emar":
                target_task.append((subject_id, hadm_id, "6"))
            elif file_name == "labevents":
                target_task.append((subject_id, hadm_id, "7"))
            elif file_name == "poe":
                target_task.append((subject_id, hadm_id, "9"))
            elif file_name == "services":
                target_task.append((subject_id, hadm_id, "8"))
                
        return list(dict.fromkeys(target_task))
    except Exception as e:
        print(f"Error processing {subject_id}-{hadm_id}: {str(e)}")
        return []

if __name__ == "__main__":
    patient_root_dir = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/my_MIMIC_IV/patients_sorted/"
    all_samples_dir = "all_samples.jsonl"
    
    # 读取所有行
    with open(all_samples_dir, 'r') as f:
        lines = f.readlines()
    
    # 使用进程池并行处理
    n_cores = cpu_count() - 1  # 保留一个核心给系统
    with Pool(n_cores) as pool:
        results = list(tqdm.tqdm(pool.imap(process_single_record, lines), total=len(lines)))
    
    # 展平结果列表
    all_tasks = [task for result in results for task in result]
    
    # 保存结果
    with open("all_sample_tasks.jsonl", "w") as f:
        for subj, hadm_id, task in all_tasks:
            f.write(json.dumps({"subject_id": subj, "hadm_id": hadm_id, "task": task}) + "\n")