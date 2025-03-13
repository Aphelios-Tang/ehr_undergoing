import os 
import pandas as pd
import json
from datetime import datetime
import jsonlines
import math
import tqdm
from joblib import Parallel, delayed
import pandas as pd
import tqdm
import csv
type_keys = {
    "radiology":{
        "note/radiology_detail": ["field_value"],
    },
    "service":{
        "hosp/services": ["curr_service"],
    },
    "omr":{
        "hosp/omr": ["result_name"],
    },
    "procedures_icd":{
        "hosp/procedures_icd": ["icd_code","icd_version"],
    },
    "diagnosis_icd":{
        "hosp/diagnoses_icd": ["icd_code","icd_version"],
        "ed/diagnosis": ["icd_code","icd_version"],
    },
    "poe":{
        "hosp/poe": ["order_type"],
    },
    "emar":{
        "hosp/emar": ["medication"],
    },
    "ed_medication":{
        "ed/pyxis": ["name"],
        "ed/medrecon": ["name"],
    },
    "microbiologyevents":{
        "hosp/microbiologyevents": ["test_name"],       
    },
    "labevents":{
        "hosp/labevents": ["itemid"],
        #"hosp/microbiologyevents": ["test_name"]        
    },
    "hcpcsevents":{
        "hosp/hcpcsevents": ["hcpcs_cd"],
    },
    "ed_triage":{
        "ed/triage": ["acuity"],
    },
    "transfer":{
        "hosp/transfers": ["eventtype"],
    },
    "icu_events": {
        "icu/ingredientevents":["itemid"],
        "icu/inputevents":["itemid"],
        "icu/outputevents":["itemid"],
        "icu/procedureevents":["itemid"],
        "icu/chartevents":["itemid"],
    },  
}
# "prescriptions":{
#         "hosp/prescriptions": ["drug"],
#     }
ROOT_DIR = "/mnt/hwfile/medai/zhangxiaoman/DATA/MIMIC-IV/mimiciv/2.2/"


for sub_key in type_keys.keys():
    result_dict = {}
    sub_path = ROOT_DIR
    sub_type_keys = type_keys[sub_key]
    # result_dict[sub_key] = {}
    for subsub_key in tqdm.tqdm(list(sub_type_keys.keys())):
        csv_path = sub_path + '/' + subsub_key + '.csv'
        #result_dict[sub_key][subsub_key]=[]
        subsubsub_key_list = sub_type_keys[subsub_key]
        combined_items = pd.DataFrame()
        with open(csv_path,'r') as f:
            items = pd.read_csv(f)
            if sub_key == "radiology":
                items = items[items["field_name"] == "exam_name"]
        combined_items = pd.concat([combined_items, items])    
        print(csv_path)
        # items = items[items["field_name"] == "exam_name"]
        # 获取 `subsubsub_key_list[0]` 列的数据
        #column_data = items[subsubsub_key_list[0]]
        #element_counts = column_data.value_counts()
        
    combined_counts = combined_items.groupby(subsubsub_key_list).size()
    sorted_combined_counts = combined_counts.sort_values(ascending=False)
    sorted_combined_counts.to_csv("./item_set/" + sub_key + ".csv", header=False)
        
    # with open("./item_set/" + sub_key + ".csv",'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(["item_name","number"])
    #     for item_name in list(result.keys()):
    #         writer.writerows([item_name,result[item_name]])
            
        