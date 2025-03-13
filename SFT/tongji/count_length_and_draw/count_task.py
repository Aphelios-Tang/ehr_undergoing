import os 
import pandas as pd
import json
import datetime
import jsonlines
from torch.utils.data import Dataset, DataLoader
import tqdm
from functools import *
import random
from utils import * # 包含数据处理的函数，xxx_item_to_free_text
import transformers
import copy
from joblib import Parallel, delayed
from collections import Counter
import itertools


SEED = 42
transformers.set_seed(SEED)
IGNORE_INDEX = -100

class MIMIC_Dataset(Dataset):
    def __init__(self, patient_root_dir, patient_id_csv):
        
        self.patient_root_dir = patient_root_dir
        self.patients_id = pd.read_csv(patient_id_csv)["Patient_ID"].tolist()

        # 遍历subjects，获取所有hadm文件列表
        self.samples = []
        self.subject_to_hadm = {}  # 新增字典，存储subject_id对应的hadm_id列表
        for subj in self.patients_id:
            hadms_dir = os.path.join(self.patient_root_dir, str(subj), "hadms")
            if not os.path.exists(hadms_dir):
                continue
            hadm_ids = []
            for fname in os.listdir(hadms_dir):
                if fname.endswith(".jsonl"):
                    hadm_id = fname.replace(".jsonl","")
                    self.samples.append((subj, hadm_id))
                    hadm_ids.append(hadm_id)
            self.subject_to_hadm[subj] = hadm_ids
        print(f"Total samples: {len(self.samples)}")
        print(f"Total patients: {len(self.patients_id)}")
    def __len__(self):
        # return len(self.patients_id)
        return len(self.samples)

    def __getitem__(self,idx):
        subject_id, hadm_id = self.samples[idx]
        hadm_file = os.path.join(self.patient_root_dir, str(subject_id), "hadms", f"{hadm_id}.jsonl")
        patient_trajectory_list = read_jsonl(hadm_file)
        target_task_v = self.process_cases(patient_trajectory_list, subject_id, hadm_id)
        
        return target_task_v
    
    def process_cases(self, patient_trajectory_list, subject_id, hadm_id, random_flag=True):
        
        # TASK = ["1_in_hospital_mortality_prediction", "2_LOS_prediction", "3_diagnosis_prediction", "4_readmission_prediction", "5_transfer_icu","6_medication_combination_prediction", "7_next_labortory_group", "8_medical_report_summerization", "9_treatment_recommendation","10_lab_results_interpretation", "11_service_prediction"]
        # 根据enumerate的item，获取file_name，判断是否是TASK中的task，如果是，加入到target_task中      
        target_task = [] # ****
        for ii, item in enumerate(patient_trajectory_list):
            file_name = item["file_name"]
            if file_name == "admissions":
                target_task.append("1_in_hospital_mortality_prediction") # finish
                target_task.append("2_LOS_prediction") # finish
                target_task.append("4_readmission_prediction") # finish
            elif file_name == "diagnoses_icd":
                target_task.append("3_diagnosis_prediction") # finish
            elif file_name == "ed":
                target_task.append("5_transfer_icu") # finish
            elif file_name == "emar": 
                target_task.append("6_medication_combination_prediction") # 开药怎么做？
            elif file_name == "labevents":
                target_task.append("7_next_labortory_group") # finish
                target_task.append("10_lab_results_interpretation") # finish
            elif file_name == "note" and "DS" in item["items"][0]["note_type"]:
                target_task.append("8_medical_report_summerization") # finish
            elif file_name == "poe":
                target_task.append("9_treatment_recommendation") # finish
            elif file_name == "services":
                target_task.append("11_service_prediction") # finish 分诊，转科，比如骨科什么的

        # 去重并保留顺序
        target_task = list(dict.fromkeys(target_task))
        # print(target_task)

        assert len(target_task)!= 0, print("NO TARGET ITEM. ERROR CASE ID: ", subject_id, hadm_id)

        # 设立flag，训练时可以随机选task，也可以选最后一个task
        if random_flag == True:
            target_task_v = random.choice(target_task)
        else:
            selected_task_num = random_flag
            filtered_task = [task for task in target_task if task.startswith(selected_task_num)]
            target_task_v = filtered_task[0]

        
        return target_task_v


if __name__ == "__main__":
    # 创建一个数据集

    dataset = MIMIC_Dataset(patient_root_dir = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/my_MIMIC_IV/patients_sorted/",patient_id_csv = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/bohao_mimic_project/supplementary_files/split/train_patients.csv")
    
    task = {
        '1_in_hospital_mortality_prediction': 0,
        '2_LOS_prediction': 0,
        '3_diagnosis_prediction': 0,
        '4_readmission_prediction': 0,
        '5_transfer_icu': 0,
        '6_medication_combination_prediction': 0,
        '7_next_labortory_group': 0,
        '8_medical_report_summerization': 0,
        '9_treatment_recommendation': 0,
        '10_lab_results_interpretation': 0,
        '11_service_prediction': 0
    }

    for i in tqdm.tqdm(range(len(dataset.samples))): 
        data = dataset[i]
        task[data] += 1
    
    print(task)

            
    