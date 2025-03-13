import os
import json
import tqdm

patient_root_dir = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/my_MIMIC_IV/patients_sorted/"
patients_id = os.listdir(patient_root_dir)
samples = []
subject_to_hadm = {}  # 新增字典，存储subject_id对应的hadm_id列表
for subj in tqdm.tqdm(patients_id):
    hadms_dir = os.path.join(patient_root_dir, str(subj), "hadms")
    if not os.path.exists(hadms_dir):
        continue
    hadm_ids = []
    for fname in os.listdir(hadms_dir):
        if fname.endswith(".jsonl"):
            hadm_id = fname.replace(".jsonl","")
            samples.append((subj, hadm_id))
            hadm_ids.append(hadm_id)
    subject_to_hadm[subj] = hadm_ids

# 保存samples为jsonl
with open("all_samples.jsonl", "w") as f:
    for subj, hadm_id in samples:
        f.write(json.dumps({"subject_id": subj, "hadm_id": hadm_id}) + "\n")

# 保存subject_to_hadm为json
with open("subject_to_hadm.json", "w") as f:
    json.dump(subject_to_hadm, f)