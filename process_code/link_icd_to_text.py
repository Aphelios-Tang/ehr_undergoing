import tqdm
import os
import json
import jsonlines
import pandas as pd
from joblib import Parallel, delayed

with open("/mnt/petrelfs/wuchaoyi/my_MIMIC_IV_2.2/process_code/disease_statistic.json",'r') as f:
    json_file = json.load(f)

with open("/mnt/petrelfs/wuchaoyi/my_MIMIC_IV_2.2/supplementary_files/hosp/d_icd_diagnoses.csv") as f:
    icd_index = pd.read_csv(f)

inflect_index = {"9":{},"10":{}}
for index in range(len(icd_index)):
    sample = icd_index.iloc[index]
    inflect_index[str(sample["icd_version"])][str(sample["icd_code"])] = sample["long_title"]

save_json = {}
for key in json_file.keys():
    disease_list = json_file[key]
    save_json[key] = []
    for icd_tuple in disease_list:
        save_json[key].append(inflect_index[str(icd_tuple[1])][str(icd_tuple[0])])
            #save_json[key] = [inflect_index[str(icd_tuple[1])][str(icd_tuple[0])]]
    # print(key)
    
with open('disease_statistic_free_text.json', 'w', encoding='utf-8') as json_file:
    json.dump(save_json, json_file, ensure_ascii=False, indent=4)
    