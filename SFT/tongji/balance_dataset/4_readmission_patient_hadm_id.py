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
from DS import *
import torch
from collections import defaultdict
import multiprocessing as mp
from functools import partial

SEED = 42
transformers.set_seed(SEED)
IGNORE_INDEX = -100

def preprocess(
    sources,
    targets,
    tokenizer,
):
   
    # print("sources",sources)
    # print("targets",targets)
    # exit()
    examples = sources + targets
    
    sources_tokenized = tokenizer(
            sources,
            return_tensors="pt",
            # max_length=tokenizer.model_max_length,
            truncation=False,
        )
    sources_ids = sources_tokenized["input_ids"][0]
    
    target_tokenized = tokenizer(
            targets,
            return_tensors="pt",
            # max_length=tokenizer.model_max_length,
            truncation=False,
        )
    target_ids = target_tokenized["input_ids"][0]
    
    # input
    examples_tokenized = tokenizer(
            examples,
            return_tensors="pt",
            # max_length=tokenizer.model_max_length,
            truncation=False,
        )
    
    combined_ids = torch.cat([sources_ids, target_ids[1:]], dim=0)
    input_ids = combined_ids
    
    # input_ids = examples_tokenized["input_ids"][0]
    
    # print(sources_ids)
    # print(target_ids)
    # print(combined_ids)
    # print(input_ids)
    
    # source_tokens = tokenizer.convert_ids_to_tokens(sources_ids)
    # target_tokens = tokenizer.convert_ids_to_tokens(target_ids)
    # combined_tokens = tokenizer.convert_ids_to_tokens(combined_ids)
    # predicted_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    # print(source_tokens[-5:])
    # print(target_tokens[-5:])
    # print(combined_tokens[-5:])
    # print(predicted_tokens[-5:])
    
    # exit()
    labels = copy.deepcopy(input_ids)
    # print(len(sources_tokenized["input_ids"][0]), len(labels))
    labels[:len(sources_tokenized["input_ids"][0])] = IGNORE_INDEX
    # print("input", input_ids)
    # print("label", labels)
    # exit()
    predicted_tokens = tokenizer.convert_ids_to_tokens(input_ids[-2:])
    # labed_tokens = tokenizer.convert_ids_to_tokens(labels[-2:])
    # print(input_ids[-5:])
    # print(labels[-5:])
    # print(tokenizer.convert_ids_to_tokens(torch.tensor([328, 1038,  281, 2305, 2])))
    # print(tokenizer.convert_ids_to_tokens(torch.tensor([2305, 2])))
    # exit()
    
    if len(input_ids) > tokenizer.model_max_length:
        input_ids = input_ids[-tokenizer.model_max_length:]
        labels = labels[-tokenizer.model_max_length:]
        # print(len(input_ids),len(labels))
    # print(dict(input_ids=input_ids, labels=labels))
    # exit()
    return dict(input_ids=input_ids, labels=labels)


class MIMIC_Dataset(Dataset):
    def __init__(self, patient_root_dir, patient_id_csv, tokenizer, context_item_window =20, icd_diagnosis_dir="/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/bohao_mimic_project/supplementary_files/hosp/d_icd_diagnoses.csv", icd_procedure_dir = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/bohao_mimic_project/supplementary_files/hosp/d_icd_procedures.csv", hosp_item = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/bohao_mimic_project/supplementary_files/hosp/d_labitems.csv", hcpcs_item = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/bohao_mimic_project/supplementary_files/hosp/d_hcpcs.csv", icu_item = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/bohao_mimic_project/supplementary_files/icu/d_items.csv"):
        
    
    # 加载一系列医疗数据字典

        print("---------------Loading ICD Diagnoisis Indexing---------------")
        self.icd_diagnosis_reflect_dict = process_icd(icd_diagnosis_dir)
        
        print("---------------Loading ICD Procedure Indexing---------------")
        self.icd_procedure_reflect_dict = process_icd(icd_procedure_dir)
        
        print("---------------Loading Hospital Item Indexing---------------")
        self.hosp_item_dict = process_hosp_item(hosp_item)
        
        print("---------------Loading HCPCS Item Indexing---------------")
        self.hcpcs_item = process_hcpcs_item(hcpcs_item)
        
        print("---------------Loading ICU Item Indexing---------------")
        self.icu_item_dict = process_icu_item(icu_item)

        # item -> free text的函数
        self.preprocess_func = {
            "patients": patients_item_to_free_text,
            "omr": omr_item_to_free_text,
            "labevents": partial(labevents_item_to_free_text, item_indexing = self.hosp_item_dict),
            "transfers": transfers_item_to_free_text,
            "poe": poe_item_to_free_text,
            "services": services_item_to_free_text,
            "note": note_item_to_free_text,
            "admissions": admissions_item_to_free_text,
            "emar": emar_item_to_free_text,
            "microbiologyevents": microbiologyevents_item_to_free_text,
            "diagnoses_icd": partial(diagnoses_icd_item_to_free_text, item_indexing = self.icd_diagnosis_reflect_dict),
            "procedures_icd": partial(procedures_icd_item_to_free_text, item_indexing = self.icd_procedure_reflect_dict),
            # "pharmacy": pharmacy_item_to_free_text, # 药房表
            "prescriptions": prescriptions_item_to_free_text,  # 处方表
            "ed": partial(ed_item_to_free_text),
            "icu": partial(icu_item_to_free_text,  item_indexing = self.icu_item_dict),
        }

        self.task_func = {
            "1_in_hospital_mortality_prediction": self.in_hospital_mortality_prediction, # finish
            "2_LOS_prediction": self.LOS_prediction,  # finish
            "3_diagnosis_prediction": self.diagnosis_prediction, # finish
            "4_readmission_prediction": self.readmission_prediction, # finish
            "5_transfer_icu": self.transfer_icu, # finish 
            "6_medication_combination_prediction": self.medication_combination_prediction, 
            "7_next_labortory_group": self.labortory_group, # finish
            # "8_medical_report_summerization": self.medical_report_summerization, # finish
            "8_service_prediction": self.service_prediction, # finish
            "9_treatment_recommendation": self.treatment_recommendation,
            # "10_lab_results_interpretation": self.lab_results_interpretation, # finish
        }
        
        self.patient_root_dir = patient_root_dir
        self.patients_hadm_task_id = pd.read_csv(patient_id_csv)

        self.tokenizer = tokenizer
        self.context_item_window = context_item_window

        # print(len(self.patients_id))

        # 遍历subjects，获取所有hadm文件列表
        self.samples = []
        self.subject_to_hadm = {}  # 新增字典，存储subject_id对应的hadm_id列表
        for index, row in self.patients_hadm_task_id.iterrows():
            subject_id = int(row['subject_id'])
            hadm_id = int(row['hadm_id'])
            task_id = str(int(row['task']))
            self.samples.append((subject_id, hadm_id, task_id))
            if subject_id not in self.subject_to_hadm:
                self.subject_to_hadm[subject_id] = []
            self.subject_to_hadm[subject_id].append(hadm_id)
        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        # return len(self.patients_id)
        return len(self.samples)

    # patient_trajectory_list 病人轨迹记录
    def __getitem__(self,idx):
        subject_id, hadm_id, task_id = self.samples[idx]
        def get_hadm_file(patient_root_dir, subject_id, hadm_id):
            # 尝试两种可能的文件名
            hadm_file1 = os.path.join(patient_root_dir, f"{subject_id}", "hadms", f"{hadm_id}.jsonl")
            hadm_file2 = os.path.join(patient_root_dir, f"{subject_id}", "hadms", f"{hadm_id}.0.jsonl")
            
            # 检查文件是否存在
            if os.path.exists(hadm_file1):
                return hadm_file1
            elif os.path.exists(hadm_file2):
                return hadm_file2
            else:
                return None
        hadm_file = get_hadm_file(self.patient_root_dir, subject_id, hadm_id)
        patient_trajectory_list = read_jsonl(hadm_file)
        free_text = self.process_cases(patient_trajectory_list, subject_id, hadm_id, task_id)
        
        # # 数据集内容是Instruction, Input, Output
        # tokens = preprocess(
        #     sources=self.tokenizer.bos_token + free_text["Input"] + SPLIT_LINE + free_text["Instruction"],
        #     targets=free_text["Output"],
        #     tokenizer=self.tokenizer
        # )
        return free_text["Output"], subject_id, hadm_id, task_id
    
    def get_all_hadm_ids_for_subject(self, subject_id):
        """
        返回指定subject_id下所有的hadm_id, task列表
        """
        return self.subject_to_hadm.get(subject_id, [])
    
    
    def get_infer_case(self, idx):
        subject_id, hadm_id, task_id = self.samples[idx]
        def get_hadm_file(patient_root_dir, subject_id, hadm_id):
            # 尝试两种可能的文件名
            hadm_file1 = os.path.join(patient_root_dir, f"{subject_id}", "hadms", f"{hadm_id}.jsonl")
            hadm_file2 = os.path.join(patient_root_dir, f"{subject_id}", "hadms", f"{hadm_id}.0.jsonl")
            
            # 检查文件是否存在
            if os.path.exists(hadm_file1):
                return hadm_file1
            elif os.path.exists(hadm_file2):
                return hadm_file2
            else:
                return None
        hadm_file = get_hadm_file(self.patient_root_dir, subject_id, hadm_id)
        patient_trajectory_list = read_jsonl(hadm_file)

        free_text = self.process_cases(patient_trajectory_list, subject_id, hadm_id, task_id)

        try:
            return {
                "question": self.tokenizer.bos_token + free_text["Input"] + SPLIT_LINE + free_text["Instruction"],
                "answer": free_text["Output"]
            } 
        except:
            return {
                "question": free_text["Input"] + SPLIT_LINE + free_text["Instruction"],
                "answer": free_text["Output"]
            }
        
    def get_item_by_subject_hadm(self, subject_id, hadm_id, task_id):
        """
        实现subject_id和hadm_id的取样
        """
        
        def get_hadm_file(patient_root_dir, subject_id, hadm_id):
            # 尝试两种可能的文件名
            hadm_file1 = os.path.join(patient_root_dir, f"{subject_id}", "hadms", f"{hadm_id}.jsonl")
            hadm_file2 = os.path.join(patient_root_dir, f"{subject_id}", "hadms", f"{hadm_id}.0.jsonl")
            
            # 检查文件是否存在
            if os.path.exists(hadm_file1):
                return hadm_file1
            elif os.path.exists(hadm_file2):
                return hadm_file2
            else:
                return None
        hadm_file = get_hadm_file(self.patient_root_dir, subject_id, hadm_id)
        patient_trajectory_list = read_jsonl(hadm_file)
        # print(patient_trajectory_list)
        free_text = self.process_cases(patient_trajectory_list, subject_id, hadm_id, task_id)

        return free_text
    
    # OK
    def in_hospital_mortality_prediction(self, patient_trajectory_list):
        """
        住院死亡预测任务
        需要输入的内容有：（24小时内的）（如存在）
        1. 病史，note-“History from past admissions”
        2. 病人的基本信息，patients
        3. 病人的电子健康记录，omr
        4. 病人的实验室检查记录，labevents
        5. 病人的转科记录，transfers
        6. 病人的手术记录，procedures_icd
        7. 病人的给药记录，emar
        8. 病人的诊断记录，diagnoses_icd
        9. 病人的icu记录，icu
        10. 病人在急诊室的记录，ed
        11. 病人的微生物检查记录，microbiologyevents
        # 12. 病人的医嘱记录，poe
        13. 病人的影像学报告，note-“RR”

        target: admission的hospital_expire_flag
        """
        patient_trajectory_list_24 = self.filter_24h_data(patient_trajectory_list) # 24h病人轨迹记录
        # print(patient_trajectory_list_24)
        transformed_input_text_list = []
        history_input_text = []
        choice_list = ["yes", "no"]
        instrucion_prompt = "Please predict the in-hospital mortality of the patient based on the above EHR information. Choose from {candidates}."
        target = "no"

        patients = []
        addention_patients = []
        for ii, item in enumerate(patient_trajectory_list):
            file_name = item["file_name"]
            if file_name == "note" and "DS" in item["items"][0]["note_type"]:
                text = get_history(item["items"][0]["text"])
                addention = get_patients_addention(item["items"][0]["text"])
                # if len(text)/4 > 4000:  # 限制历史记录的长度
                #     text = text[:4000*4]
                history_input_text.append(text)
                addention_patients.append(addention)
            elif file_name =="patients":
                patients.append(self.preprocess_func[file_name](item))

        history_and_patients = patients + addention_patients + history_input_text

        for ii, item in enumerate(patient_trajectory_list_24):
            file_name = item["file_name"]
            # 如果是历史记录，直接加入 history_input_text，需要单独存放，拼接在最前面
            if file_name == "note" and "RR" in item["items"][0]["note_type"]:
                rr = item["items"][0]["text"]
                rr_text = "Radiology Report: " + rr
                transformed_input_text_list.append(rr_text)
            elif file_name == "admissions":
                hospital_expire_flag = item["items"][0]["hospital_expire_flag"]
                if hospital_expire_flag == 1:
                    target = "yes"
            elif file_name in ["omr", "labevents", "transfers", "procedures_icd", "emar", "diagnoses_icd", "icu", "ed", "microbiologyevents"]:
                transformed_input_text_list.append(self.preprocess_func[file_name](item))
        Instruction = instrucion_prompt.format(candidates = ", ".join(choice_list))
        try:
            Output = target + self.tokenizer.eos_token
        except:
            Output = target
        
        

        return Instruction, history_and_patients, transformed_input_text_list, Output
    

    def LOS_prediction(self, patient_trajectory_list):
        """
        住院时间预测任务
        需要输入的内容有：（24小时内的）（如存在）
        1. 病史，note-“History from past admissions”
        2. 病人的基本信息，patients
        3. 病人的电子健康记录，omr
        4. 病人的实验室检查记录，labevents
        5. 病人的转科记录，transfers
        6. 病人的手术记录，procedures_icd
        7. 病人的给药记录，emar
        8. 病人的诊断记录，diagnoses_icd
        9. 病人的icu记录，icu
        10. 病人在急诊室的记录，ed
        11. 病人的微生物检查记录，microbiologyevents
        12. 病人的医嘱记录，poe
        13. 病人的影像学报告，note-“RR”

        target: admission的hospital_expire_flag
        """
        patient_trajectory_list_24 = self.filter_24h_data(patient_trajectory_list) # 24h病人轨迹记录
        transformed_input_text_list = []
        history_input_text = [] 
        choice_list = ["yes", "no"]
        instrucion_prompt = "Please predict if the length of stay of the patient will exceed 7 days based on the above EHR information. Choose from {candidates}."
        target = "no"

        patients = []
        addention_patients = []
        for ii, item in enumerate(patient_trajectory_list):
            file_name = item["file_name"]
            if file_name == "note" and "DS" in item["items"][0]["note_type"]:
                text = get_history(item["items"][0]["text"])
                addention = get_patients_addention(item["items"][0]["text"])
                # if len(text)/4 > 4000:  # 限制历史记录的长度
                #     text = text[:4000*4]
                history_input_text.append(text)
                addention_patients.append(addention)
            elif file_name =="patients":
                patients.append(self.preprocess_func[file_name](item))

        history_and_patients = patients + addention_patients + history_input_text


        for ii, item in enumerate(patient_trajectory_list_24):
            file_name = item["file_name"]
            if file_name == "note" and "RR" in item["items"][0]["note_type"]:
                rr = item["items"][0]["text"]
                rr_text = "Radiology Report: " + rr
                transformed_input_text_list.append(rr_text)
            elif file_name == "admissions":
                admittime_str = item["items"][0]["admittime"]
                dischtime_str = item["items"][0]["dischtime"]
                admittime = datetime.datetime.strptime(admittime_str, "%Y-%m-%d %H:%M:%S")
                dischtime = datetime.datetime.strptime(dischtime_str, "%Y-%m-%d %H:%M:%S")
                # 计算住院时间，单位是精确到天数
                hosp_day = (dischtime - admittime).days
                if hosp_day > 7:
                    target = "yes"
            elif file_name in ["omr", "labevents", "transfers", "procedures_icd", "emar", "diagnoses_icd", "icu", "ed", "microbiologyevents"]:
                transformed_input_text_list.append(self.preprocess_func[file_name](item))
        Instruction = instrucion_prompt.format(candidates = ", ".join(choice_list))
        try:
            Output = target + self.tokenizer.eos_token
        except:
            Output = target
        return Instruction, history_and_patients, transformed_input_text_list, Output
    
    # OK
    def readmission_prediction(self, patient_trajectory_list):
        """
        再入院预测任务：
        需要输入的内容有：（完整时间）（如存在）
        1. 病史，note-“History from past admissions”
        2. 病人的基本信息，patients
        3. 病人的电子健康记录，omr
        4. 病人的实验室检查记录，labevents
        5. 病人的转科记录，transfers
        6. 病人的手术记录，procedures_icd
        7. 病人的给药记录，emar
        8. 病人的诊断记录，diagnoses_icd
        9. 病人的icu记录，icu
        10. 病人在急诊室的记录，ed
        11. 病人的微生物检查记录，microbiologyevents
        12. 病人的医嘱记录，poe
        13. 病人的影像学报告，note-“RR”

        target: next admission的hadm_id 是否存在
        """
        def get_hadm_file(patient_root_dir, subject_id, hadm_id):
            # 尝试两种可能的文件名
            hadm_file1 = os.path.join(patient_root_dir, f"{subject_id}", "hadms", f"{hadm_id}.jsonl")
            hadm_file2 = os.path.join(patient_root_dir, f"{subject_id}", "hadms", f"{hadm_id}.0.jsonl")
            
            # 检查文件是否存在
            if os.path.exists(hadm_file1):
                return hadm_file1
            elif os.path.exists(hadm_file2):
                return hadm_file2
            else:
                return None
        

        def get_hadmid_admittime(subject_id, hadm_id):
            hadm_file = get_hadm_file(self.patient_root_dir, subject_id, hadm_id)
            # hadm_file = os.path.join(self.patient_root_dir, str(subject_id), "hadms", f"{hadm_id}.jsonl")
            patient_trajectory_list = read_jsonl(hadm_file)
            for item in patient_trajectory_list:
                if item["file_name"] == "admissions":
                    admittime_str = item["items"][0]["admittime"]
                    admittime = datetime.datetime.strptime(admittime_str, "%Y-%m-%d %H:%M:%S")
                    return admittime
            return None


        # 需要完整时间
        transformed_input_text_list = []
        history_input_text = [] 
        choice_list = ["yes", "no"]
        instrucion_prompt = "Please predict whether the patient will be readmitted based on the above EHR information. Choose from {candidates}."

        patients = []
        addention_patients = []
    
        for ii, item in enumerate(patient_trajectory_list):
            file_name = item["file_name"]
            if file_name == "note" and "DS" in item["items"][0]["note_type"]:
                text = get_history(item["items"][0]["text"])
                addention = get_patients_addention(item["items"][0]["text"])
                # if len(text)/4 > 4000:  # 限制历史记录的长度
                #     text = text[:4000*4]
                history_input_text.append(text)
                addention_patients.append(addention)
            elif file_name =="patients":
                patients.append(self.preprocess_func[file_name](item))
            elif file_name == "note" and "RR" in item["items"][0]["note_type"]:
                rr = item["items"][0]["text"]
                rr_text = "Radiology Report: " + rr
                transformed_input_text_list.append(rr_text)
            elif file_name == "admissions":
                subject_id = item["items"][0]["subject_id"]
                all_hadm_ids = self.get_all_hadm_ids_for_subject(subject_id)
                this_admittime = item["items"][0]["admittime"]
                this_admittime = datetime.datetime.strptime(this_admittime, "%Y-%m-%d %H:%M:%S")
                target = "no"
                # 遍历所有hadm_id，如果存在一个hadm_id的admittime大于this_admittime，那么target为yes
                for hadm_id in all_hadm_ids:
                    if get_hadmid_admittime(subject_id, hadm_id) > this_admittime:
                        target = "yes"
                        break 
            elif file_name in ["omr", "labevents", "transfers", "procedures_icd", "emar", "diagnoses_icd", "icu", "ed", "microbiologyevents"]:
                transformed_input_text_list.append(self.preprocess_func[file_name](item))
        Instruction = instrucion_prompt.format(candidates = ", ".join(choice_list))
        history_and_patients = patients + addention_patients + history_input_text
        try:
            Output = target + self.tokenizer.eos_token
        except:
            Output = target
        return Instruction, history_and_patients, transformed_input_text_list, Output

    # OK
    def diagnosis_prediction(self, patient_trajectory_list):
        """
        诊断预测任务：
        需要输入的内容有：（24小时内的）（如存在）
        手术，给药，icu肯定在诊断之后，所以不需要考虑
        1. 病史，note-“History from past admissions”
        2. 病人的基本信息，patients
        3. 病人的电子健康记录，omr
        4. 病人的实验室检查记录，labevents
        5. 病人在急诊室的记录，ed
        6. 病人的微生物检查记录，microbiologyevents
        7. 病人的影像学报告，note-“RR”

        target: admission的hospital_expire_flag
        """

        def get_target_items(self, item, key):
            output = [_[key] for _ in item["items"]]
            output = list(set(output))
            return output


        def make_output(self, item, key=None):
            if key is None:
                output = item
            else:
                output = [_[key] for _ in item["items"]]
            output = list(set(output))
            try:
                return ", ".join(output)+self.tokenizer.eos_token
            except:
                return ", ".join(output)

        choice_list = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/bohao_mimic_project/SFT/item_similarity/diagnosis_icd.csv"
        # patient_trajectory_list_24 = filter_24h_data(patient_trajectory_list) # 24h病人轨迹记录
        transformed_input_text_list = []
        history_input_text = [] 
        patients = []
        addention_patients = []
        instrucion_prompt = "Please predict the diagnosis of the patient based on the above EHR information. Choose from {candidates}."
        for ii, item in enumerate(patient_trajectory_list):
            file_name = item["file_name"]
            # 如果是历史记录，直接加入 history_input_text，需要单独存放，拼接在最前面
            if file_name == "note" and "DS" in item["items"][0]["note_type"]:
                text = get_history(item["items"][0]["text"])
                addention = get_patients_addention(item["items"][0]["text"])
                # if len(text)/4 > 4000:  # 限制历史记录的长度
                #     text = text[:4000*4]
                history_input_text.append(text)
                addention_patients.append(addention)
            elif file_name =="patients":
                patients.append(self.preprocess_func[file_name](item))
            elif file_name == "note" and "RR" in item["items"][0]["note_type"]:
                rr = item["items"][0]["text"]
                rr_text = "Radiology Report: " + rr
                transformed_input_text_list.append(rr_text)
            elif file_name in ["omr", "labevents", "ed", "microbiologyevents"]:
                transformed_input_text_list.append(self.preprocess_func[file_name](item))
            elif file_name == "diagnoses_icd":
                target_items_id = self.get_target_items(item, "icd_code")
                candidate_items, target_items = similarity_sample(target_items_id, choice_list)
                Instruction = instrucion_prompt.format(candidates = ", ".join(candidate_items))
                Output = self.make_output(target_items)
        history_and_patients = patients + addention_patients + history_input_text
        return Instruction, history_and_patients, transformed_input_text_list, Output
    
    # OK
    def transfer_icu(self, patient_trajectory_list):
        """
        转科预测任务：
        需要输入的内容有：（24小时内的）（如存在）
        1. 病史，note-“History from past admissions”
        2. 病人的基本信息，patients
        3. 病人的电子健康记录，omr
        4. 病人的实验室检查记录，labevents
        5. 病人在急诊室的记录，ed
        target: next admission的admission_type 是否为transfer
        """
        patient_trajectory_list_24 = self.filter_24h_data(patient_trajectory_list) # 24h病人轨迹记录
        transformed_input_text_list = []
        history_input_text = []
        choice_list = ["yes", "no"]
        instrucion_prompt = "Please predict whether the patient will be transferred to ICU based on the above EHR information. Choose from {candidates}."

        target = "no"

        patients = []
        addention_patients = []
        for ii, item in enumerate(patient_trajectory_list):
            file_name = item["file_name"]
            if file_name == "note" and "DS" in item["items"][0]["note_type"]:
                text = get_history(item["items"][0]["text"])
                addention = get_patients_addention(item["items"][0]["text"])
                # if len(text)/4 > 4000:  # 限制历史记录的长度
                #     text = text[:4000*4]
                history_input_text.append(text)
                addention_patients.append(addention)
            elif file_name =="patients":
                patients.append(self.preprocess_func[file_name](item))

        history_and_patients = patients + addention_patients + history_input_text

        for ii, item in enumerate(patient_trajectory_list_24):
            file_name = item["file_name"]
            # 如果是历史记录，直接加入 history_input_text，需要单独存放，拼接在最前面
            if file_name in ["omr", "labevents", "ed"]:
                transformed_input_text_list.append(self.preprocess_func[file_name](item))
            elif file_name == "icu":
                target = "yes"
        Instruction = instrucion_prompt.format(candidates = ", ".join(choice_list))
        try:
            Output = target + self.tokenizer.eos_token
        except:
            Output = target
        return Instruction, history_and_patients, transformed_input_text_list, Output
    
    # OK
    def medication_combination_prediction(self, patient_trajectory_list):
        """
        药物组合预测任务：
        需要输入的内容有：（24小时内的）（如存在）
        1. 病史，note-“History from past admissions”
        2. 病人的基本信息，patients
        3. 病人的电子健康记录，omr
        4. 病人的实验室检查记录，labevents
        5. 病人在急诊室的记录，ed
        6. 病人的手术记录，procedures_icd
        7. 病人的诊断记录，diagnoses_icd
        target: emar的medication
        """
        def get_target_items(self, item, key):
            output = [_[key] for _ in item["items"]]
            output = list(set(output))
            return output

        def make_output(self, item, key=None):
            if key is None:
                output = item
            else:
                output = [_[key] for _ in item["items"]]
            output = list(set(output))
            try:
                return ", ".join(output)+self.tokenizer.eos_token
            except:
                return ", ".join(output)
        choice_list = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/bohao_mimic_project/SFT/item_similarity/emar.csv"
        # patient_trajectory_list_24 = filter_24h_data(patient_trajectory_list)
        transformed_input_text_list = []
        history_input_text = []
        patients = []
        addention_patients = []
        instrucion_prompt = "Please predict the medication combination of the patient based on the above EHR information. Choose from {candidates}."
        for ii, item in enumerate(patient_trajectory_list):
            file_name = item["file_name"]
            # 如果是历史记录，直接加入 history_input_text，需要单独存放，拼接在最前面
            if file_name == "note" and "DS" in item["items"][0]["note_type"]:
                text = get_history(item["items"][0]["text"])
                addention = get_patients_addention(item["items"][0]["text"])
                # if len(text)/4 > 4000:  # 限制历史记录的长度
                #     text = text[:4000*4]
                history_input_text.append(text)
                addention_patients.append(addention)
            elif file_name =="patients":
                patients.append(self.preprocess_func[file_name](item))
            elif file_name in ["omr", "labevents", "ed", "procedures_icd", "diagnoses_icd"]:
                transformed_input_text_list.append(self.preprocess_func[file_name](item))
            elif file_name == "emar":
                target_items_id = self.get_target_items(item, "medication")
                candidate_items, target_items = similarity_sample(target_items_id, choice_list)
                Instruction = instrucion_prompt.format(candidates = ", ".join(candidate_items))
                Output = self.make_output(target_items)
        history_and_patients = patients + addention_patients + history_input_text
        return Instruction, history_and_patients, transformed_input_text_list, Output

    
    # OK
    def labortory_group(self, patient_trajectory_list):
        """
        实验室检查预测任务：
        需要输入的内容有：（24小时内的）（如存在）
        1. 病史，note-“History from past admissions”
        2. 病人的基本信息，patients
        3. 病人的电子健康记录，omr
        4. 病人在急诊室的记录，ed
        5. 病人的医嘱记录，poe
        6. 病人的实验室检查记录，labevents(已经检查的部分)
        target: 实验室检查的group（剩余的部分）
        """

        def get_target_items(self, item, key):
            output = [_[key] for _ in item["items"]]
            output = list(set(output))
            return output

        def make_output(self, item, key=None):
            if key is None:
                output = item
            else:
                output = [_[key] for _ in item["items"]]
            output = list(set(output))
            try:
                return ", ".join(output)+self.tokenizer.eos_token
            except:
                return ", ".join(output)
        choice_list = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/bohao_mimic_project/SFT/item_similarity/labevents.csv"
        # patient_trajectory_list_24 = self.filter_24h_data(patient_trajectory_list)
        transformed_input_text_list = []
        history_input_text = []
        patients = []
        addention_patients = []
        instrucion_prompt = "Please predict what laboratory eximination the patient will have based on the above EHR information. Choose from {candidates}."
        num_labevents = 0
        for ii, item in enumerate(patient_trajectory_list):
            if item["file_name"] == "labevents":
                num_labevents += 1
        if num_labevents > 1: 
            target_labevents_num = random.randint(1, num_labevents)
        else:
            target_labevents_num = 1
        current_labevents_num = 0
        for ii, item in enumerate(patient_trajectory_list):
            file_name = item["file_name"]
            # 如果是历史记录，直接加入 history_input_text，需要单独存放，拼接在最前面
            if file_name == "note" and "DS" in item["items"][0]["note_type"]:
                text = get_history(item["items"][0]["text"])
                addention = get_patients_addention(item["items"][0]["text"])
                # if len(text)/4 > 4000:  # 限制历史记录的长度
                #     text = text[:4000*4]
                history_input_text.append(text)
                addention_patients.append(addention)
            elif file_name =="patients":
                patients.append(self.preprocess_func[file_name](item))
            elif file_name in ["omr", "ed"]:
                transformed_input_text_list.append(self.preprocess_func[file_name](item))
            elif file_name == "labevents":
                current_labevents_num += 1
                if current_labevents_num < target_labevents_num:
                    transformed_input_text_list.append(self.preprocess_func[file_name](item))
                else:
                    target_items_id = self.get_target_items(item, "itemid")
                    candidate_items, target_items = similarity_sample(target_items_id, choice_list)
                    Instruction = instrucion_prompt.format(candidates = ", ".join(candidate_items))
                    Output = self.make_output(target_items)
        history_and_patients = patients + addention_patients + history_input_text
        return Instruction, history_and_patients, transformed_input_text_list, Output
    
    # OK
    def lab_results_interpretation(self, patient_trajectory_list):
        """
        实验室检查结果解释任务：
        需要输入的内容有：
        1. 病人的实验室检查记录，labevents（去除flag）
        target: 实验室检查的group
        """
        def labevents_item_to_free_text_lab(item, item_indexing):
            item_list = item["items"]
            start_time = item_list[0]["charttime"]
            end_time = item_list[-1]["charttime"]
            prompt = "Event_type: Labevent, Time: {start_time} -> {end_time}\n\n".format(start_time=start_time, end_time=end_time)
            
            chart_item = []
            chart_prompt = "{item_name}: {valuenum} {valueuom}  ({ref_range_lower}, {ref_range_upper})"
            for item in item_list:
                item_name = item_indexing[safe_read(item["itemid"])]
                valuenum = safe_read(item["valuenum"])
                valueuom = safe_read(item["valueuom"]) 
                ref_range_lower = safe_read(item["ref_range_lower"]) 
                ref_range_upper = safe_read(item["ref_range_upper"]) 
                # flag = safe_read(item["flag"]) 
                # comments = safe_read(item["comments"]) 
                # flag = "normal" if flag =="" else flag
                if valuenum != "":
                    chart_item.append(chart_prompt.format(item_name=item_name,valuenum=valuenum, valueuom= valueuom, ref_range_lower = ref_range_lower, ref_range_upper=ref_range_upper))
            
            chart_item_str = "\n".join(chart_item)
            
            prompt = prompt + chart_item_str
            
            return prompt
        def labevents_item_to_free_text_answer(item, item_indexing):
            item_list = item["items"]
            chart_item = []
            chart_prompt = "{item_name}: {flag}"
            for item in item_list:
                item_name = item_indexing[safe_read(item["itemid"])]
                flag = safe_read(item["flag"]) 
                valuenum = safe_read(item["valuenum"])
                flag = "normal" if flag =="" else flag
                if valuenum != "":
                    chart_item.append(chart_prompt.format(item_name=item_name, flag=flag))
            
            chart_item_str = "\n".join(chart_item)
            
            prompt = chart_item_str
            
            return prompt
        
        transformed_input_text_list = []
        target = []
        instrucion_prompt = "Please interpret the lab results of the patient based on the above EHR information."
        
        for ii, item  in enumerate(patient_trajectory_list):
            if item["file_name"] == "labevents":
                transformed_input_text_list.append(labevents_item_to_free_text_lab(item, self.hosp_item_dict))
                target.append(labevents_item_to_free_text_answer(item, self.hosp_item_dict))
        Instruction = instrucion_prompt
        Output = "\n".join(target)
        history_and_patients = []
        return Instruction, history_and_patients, transformed_input_text_list, Output
    
    # OK
    def service_prediction(self, patient_trajectory_list):
        """
        服务预测任务：
        需要输入的内容有：service之前的所有内容
        1. 病史，note-“History from past admissions”
        2. 病人的基本信息，patients
        3. 病人的电子健康记录，omr
        4. 病人的实验室检查记录，labevents
        5. 病人的转科记录，transfers
        6. 病人的手术记录，procedures_icd
        7. 病人的给药记录，emar
        8. 病人的诊断记录，diagnoses_icd
        9. 病人的icu记录，icu
        10. 病人在急诊室的记录，ed
        11. 病人的微生物检查记录，microbiologyevents
        12. 病人的影像学报告，note-“RR”
        13. 病人的医嘱记录，poe
        target: service的curr_service
        """
        transfers_input_text_list = []
        history_input_text = []
        patients = []
        addention_patients = []
        choice_list = ["MED", "CMED", "SURG","OMED","ORTHO","NMED", "OBS","NSURG","CSURG","VSURG", "PSYCH", "TRAUM","GYN", "GU", "TSURG", "PSURG","ENT", "EYE","DENT"]
        instrucion_prompt = "Please predict the service of the patient based on the above EHR information. Choose from {candidates}."
        chart_item = []
        chart_prompt = "{curr_service}"
        for ii, item in enumerate(patient_trajectory_list):
            file_name = item["file_name"]
            if file_name == "note" and "DS" in item["items"][0]["note_type"]:
                text = get_history(item["items"][0]["text"])
                addention = get_patients_addention(item["items"][0]["text"])
                # if len(text)/4 > 4000:  # 限制历史记录的长度
                #     text = text[:4000*4]
                history_input_text.append(text)
                addention_patients.append(addention)
            elif file_name =="patients":
                patients.append(self.preprocess_func[file_name](item))
            elif file_name in ["omr", "labevents", "transfers", "procedures_icd", "emar", "diagnoses_icd", "icu", "ed", "microbiologyevents"]:
                transfers_input_text_list.append(self.preprocess_func[file_name](item))
            elif file_name == "services":
                item_list = item["items"]
                for item in item_list:
                    curr_service = item["curr_service"]
                    chart_item.append(chart_prompt.format(curr_service=curr_service))
                target = ",".join(chart_item)
                break
        Instruction = instrucion_prompt.format(candidates = ", ".join(choice_list))
        history_and_patients = patients + addention_patients + history_input_text
        try:
            Output = target + self.tokenizer.eos_token
        except:
            Output = target
        return Instruction, history_and_patients, transfers_input_text_list, Output
    
    # 这个目前也有问题，这里是给出poe里面的答案
    # 但是实际上的治疗不应该是这样
    # OK
    def treatment_recommendation(self, patient_trajectory_list):  
        """
        治疗建议任务：
        需要输入的内容有：poe之前的以下所有内容
        1. 病史，note-“History from past admissions”
        2. 病人的基本信息，patients
        3. 病人的电子健康记录，omr
        4. 病人的实验室检查记录，labevents
        5. 病人的转科记录，transfers
        6. 病人的手术记录，procedures_icd
        7. 病人的给药记录，emar
        8. 病人的诊断记录，diagnoses_icd
        9. 病人的icu记录，icu
        10. 病人在急诊室的记录，ed
        11. 病人的微生物检查记录，microbiologyevents
        12. 病人的影像学报告，note-“RR”
        输出：poe的order_type
        """
        transfers_input_text_list = []
        history_input_text = []
        patients = []
        addention_patients = []
        choice_list = ["Medications", "Lab", "General Care", "ADT orders", "IV therapy", "Nutrition", "Radiology", "Consults", "Blood Bank", "Respiratory", "Cardiology", "TPN", "Critical Care", "Hemodialysis", "Neurology", "OB"]
        instrucion_prompt = "Please predict the POE recommendation of the patient based on the above EHR information. Choose from {candidates}."
        num_poe = 0
        for ii, item in enumerate(patient_trajectory_list):
            if item["file_name"] == "poe":
                num_poe += 1
        if num_poe > 1:
            target_poe = random.randint(1, num_poe)
        else:
            target_poe = 1
        current_poe = 0
        for ii, item in enumerate(patient_trajectory_list):
            file_name = item["file_name"]
            if file_name == "note" and "DS" in item["items"][0]["note_type"]:
                text = get_history(item["items"][0]["text"])
                addention = get_patients_addention(item["items"][0]["text"])
                # if len(text)/4 > 4000:  # 限制历史记录的长度
                #     text = text[:4000*4]
                history_input_text.append(text)
                addention_patients.append(addention)
            elif file_name =="patients":
                patients.append(self.preprocess_func[file_name](item))
            elif file_name in ["omr", "labevents", "transfers", "procedures_icd", "emar", "diagnoses_icd", "icu", "ed", "microbiologyevents"]:
                transfers_input_text_list.append(self.preprocess_func[file_name](item))
            elif file_name == "poe":
                current_poe += 1
                if current_poe < target_poe:
                    transfers_input_text_list.append(self.preprocess_func[file_name](item))
                else:
                    target = item["items"][0]["order_type"]
                    break
        Instruction = instrucion_prompt.format(candidates = ", ".join(choice_list))
        history_and_patients = patients + addention_patients + history_input_text
        try:
            Output = target + self.tokenizer.eos_token
        except:
            Output = target
        return Instruction, history_and_patients, transfers_input_text_list, Output
                

    
    # 处理患者的病例轨迹，形成输入模型的数据，包含Instruction, Input, Output
    def process_cases(self, patient_trajectory_list, subject_id, hadm_id, task_id="1"):

        Instruction_prompt = "{Instruction}"
        output_prompt = "{output}"
           
        TASK = ["1_in_hospital_mortality_prediction", "2_LOS_prediction", "3_diagnosis_prediction", "4_readmission_prediction", "5_transfer_icu","6_medication_combination_prediction", "7_next_labortory_group", "8_service_prediction", "9_treatment_recommendation"]
          
        task_0 = [task for task in TASK if task.startswith(task_id)]
        try:
            target_task_v = task_0[0]
        except:
            return False

        # print("target_task_v: ", target_task_v)      

        # 按照任务类型，获取对应的instruction，input，output
        Instruction, history_and_patients, transformed_input_text_list, Output = self.task_func[target_task_v](patient_trajectory_list)
        history_and_patients_text = SPLIT_LINE.join(history_and_patients) + SPLIT_LINE
        other_len = len(Instruction + history_and_patients_text + Output)/4
        while len(SPLIT_LINE.join(transformed_input_text_list) + SPLIT_LINE)/4 + other_len > 4400 and len(transformed_input_text_list) > 0:
            transformed_input_text_list.pop(0)
        
        transformed_input_text_list = history_and_patients + transformed_input_text_list
        
        return {
            "Instruction": Instruction_prompt.format(Instruction=Instruction),
            "Input": SPLIT_LINE.join(transformed_input_text_list) + SPLIT_LINE,
            "Output": output_prompt.format(output=Output),
        }
        
        #return 0
   


    #
    def get_target_items(self, item, key):
        output = [_[key] for _ in item["items"]]
        output = list(set(output))
        return output


    def make_output(self, item, key=None):
        if key is None:
            output = item
        else:
            output = [_[key] for _ in item["items"]]
        output = list(set(output))
        try:
            return ", ".join(output)+self.tokenizer.eos_token
        except:
            return ", ".join(output)
    
    def filter_24h_data(self, patient_trajectory_list):
        """
        过滤 patient_trajectory_list，只保留入院24小时内的数据。
        """
        TIME_FIELDS = [
            'admittime', 'charttime', 'deathtime', 'dischtime', 'edouttime', 
            'edregtime', 'entertime', 'intime', 'ordertime', 'outtime', 
            'starttime', 'stoptime', 'storetime', 'transfertime', 'verifiedtime', 
            'storedate', 'chartdate'
        ]

        admission_time = None
        # 在患者轨迹中查找 admissions 项，获取入院时间
        for item in patient_trajectory_list:
            if item.get("file_name","") == "admissions":
                try:
                    admission_str = item["items"][0].get("admittime","")
                    admission_time = pd.to_datetime(admission_str)
                    break
                except Exception as e:
                    # print(f"无法解析 admittime: {e}")
                    pass

        if not admission_time:
            # 若无法获取入院时间，默认不过滤
            return patient_trajectory_list

        # print("admission_time:", admission_time)
        filtered_trajectory = []
        cutoff_time = admission_time + pd.Timedelta(hours=24)

        for item in patient_trajectory_list:
            item_list = item.get("items", [])
            # 筛选 item_list 中时间在 24 小时内的
            valid_events = []
            for evt in item_list:
                earliest_time = None
                for f in TIME_FIELDS:
                    t_str = evt.get(f, "")
                    if not t_str:
                        continue
                    try:
                        t_dt = pd.to_datetime(t_str)
                        if earliest_time is None or t_dt < earliest_time:
                            earliest_time = t_dt
                    except Exception as e:
                        # print(f"无法解析时间字段 {f} 的值: {t_str}，错误: {e}")
                        pass
                # 确保 earliest_time 在入院时间和截止时间之间
                if earliest_time is not None and earliest_time <= cutoff_time:
                    valid_events.append(evt)
            
            # 若筛选后仍有数据，则加入
            if len(valid_events) > 0:
                new_item = dict(item)
                new_item["items"] = valid_events
                filtered_trajectory.append(new_item)

        return filtered_trajectory

if __name__ == "__main__":
    # 创建一个数据集
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/mmedlm_model/MMed-Llama3-8B/MMed-Llama-3-8B",
        model_max_length=2048,
        use_fast=False,
        trust_remote_code=True
    )
    dataset = MIMIC_Dataset(patient_root_dir = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/my_MIMIC_IV/patients_sorted/",patient_id_csv = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/bohao_mimic_project/supplementary_files/new_split/train_task4_mini.csv",tokenizer= tokenizer)
    yes_records = []
    no_records = []
    batch_stats = defaultdict(int)
    flag = 0
    for idx in tqdm.tqdm(range(len(dataset))):
        
        yes_or_no, subject_id, hadm_id, task_id = dataset[idx]
        
        record = {
                    "patient_id": subject_id,
                    "hadm_id": hadm_id,
                    "task": task_id
                }
        if yes_or_no.startswith("yes"):
            yes_records.append(record)
            flag = 1
            batch_stats[flag] += 1
        elif yes_or_no.startswith("no"):
            no_records.append(record)
            flag = 0
            batch_stats[flag] += 1
    print(f"yes: {batch_stats[1]}, no: {batch_stats[0]}")
    for filename, records in [("4_yes.jsonl", yes_records),
                            ("4_no.jsonl", no_records)]:
        with open(filename, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
