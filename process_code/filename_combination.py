
# 处理每个combined.jsonl文件，将其中的数据按照file_name进行分组，然后保存到filename_combined.jsonl文件中。
import tqdm
import os
import json
import jsonlines
import pandas as pd
from joblib import Parallel, delayed

subject_dir = "/Users/tangbohao/Desktop/mimic/test/"
patients = os.listdir(subject_dir)


    
def save_jsonl(data_list, file_name):
    
    with jsonlines.open(file_name, mode='w') as writer:
        writer.write_all(data_list)
        
def process_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    # Convert the list to DataFrame
    # df = pd.DataFrame(data)

    organized_data = []
    #current_hadm_id = None
    current_file_name = None

    for index, row in enumerate(data):
        #hadm_id = row['hadm_id']
        file_name = row['file_name']
        try:
            hadm_id = row['hadm_id']
            if pd.isna(hadm_id):
                hadm_id = None
        except:
            hadm_id = None
            
        if file_name != current_file_name:
            if index != 0:
                organized_data.append(item)
            current_file_name = file_name
            item = {
                "file_name": file_name,
                "hadm_id": hadm_id,
                "items":[row],
            }
        else:
            item["items"].append(row)
    organized_data.append(item)
    #print(organized_data)
    save_jsonl(data_list = organized_data, file_name= file_path.replace("combined.jsonl","filename_combined.jsonl"))
    

        
def process_subject(subject):
    process_jsonl(subject_dir + '/' + subject + "/combined.jsonl")
    
if __name__ == "__main__":
    # 使用 parfor 的方式处理
    Parallel(n_jobs=-1)(delayed(process_subject)(subject) for subject in tqdm.tqdm(patients))
