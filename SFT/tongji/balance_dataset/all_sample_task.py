import os
import json
import tqdm

def read_jsonl(jsonl_dir):
    data_list = []
    with open(jsonl_dir, 'r') as file:
        for line in file:
            data_list.append(json.loads(line))
    return data_list

patient_root_dir = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/my_MIMIC_IV/patients_sorted/"
all_samples_dir = "all_samples.jsonl"
# 读取all_samples.jsonl文件
all_tasks = []
for line in tqdm.tqdm(open(all_samples_dir)):
    data = json.loads(line.strip())
    subject_id = data["subject_id"]
    hadm_id = data["hadm_id"]
    hadm_file = os.path.join(patient_root_dir, str(subject_id), "hadms", f"{hadm_id}.jsonl")
    patient_trajectory_list = read_jsonl(hadm_file)
    target_task = []
    for ii, item in enumerate(patient_trajectory_list):
        file_name = item["file_name"]
        if file_name == "admissions":
            target_task.append((subject_id, hadm_id, "1"))
            target_task.append((subject_id, hadm_id, "2"))
            target_task.append((subject_id, hadm_id, "4"))
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
    target_task = list(dict.fromkeys(target_task))
    for task in target_task:
        all_tasks.append(task)

# 保存all_tasks为jsonl
with open("all_sample_tasks.jsonl", "w") as f:
    for subj, hadm_id, task in all_tasks:
        f.write(json.dumps({"subject_id": subj, "hadm_id": hadm_id, "task": task}) + "\n")