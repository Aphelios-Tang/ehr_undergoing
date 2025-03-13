# hosp.py
import os
import pandas as pd
import json
import datetime
import jsonlines
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(obj, bytes):
            return obj.decode('utf-8')
        return super(MyEncoder, self).default(obj)

def save_item_whole(subject_dict, source, root_dir):
    """
    批量保存每个subject_id的数据到对应的jsonl文件中。
    """
    for subject_id, items in tqdm.tqdm(subject_dict.items(), desc="Saving JSONL files"):
        patient_dir = os.path.join(root_dir, str(subject_id))
        os.makedirs(patient_dir, exist_ok=True)
        
        jsonl_path = os.path.join(patient_dir, f'{source}.jsonl')
        with jsonlines.open(jsonl_path, 'w') as f:
            for item in items:
                f.write(item)

def process_csv(csv_path, source, file_name):
    """
    处理单个CSV文件，将数据按subject_id分类。
    """
    subject_dict = defaultdict(list)
    chunksize = 50000  # 根据内存大小调整
    try:
        for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
            for _, row in chunk.iterrows():
                sample = row.to_dict()
                sample['file_name'] = file_name
                subject_id = sample.get("subject_id")
                if subject_id is not None:
                    subject_dict[subject_id].append(sample)
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
    return subject_dict

def merge_subject_dicts(main_dict, new_dict):
    """
    合并多个subject_dict到主dict中。
    """
    for subject_id, items in new_dict.items():
        main_dict[subject_id].extend(items)

def main():
    root_path = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/decompressed_MIMIC_IV/"
    source = "hosp"
    
    csv_dir = os.path.join(root_path, source)
    output_root_dir = '/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/my_MIMIC_IV/patients/'
    
    # 获取所有需要处理的CSV文件，排除不需要的文件
    csv_list = [f for f in os.listdir(csv_dir) if f.endswith(".csv") and f not in [
        "d_hcpcs.csv", "d_icd_diagnoses.csv", "d_icd_procedures.csv",
        "d_labitems.csv", "poe_detail.csv", "emar_detail.csv", "provider.csv"
    ]]
    print(f"Total CSV files to process: {len(csv_list)}")
    
    final_subject_dict = defaultdict(list)
    
    # 使用ProcessPoolExecutor进行并行处理
    with ProcessPoolExecutor(max_workers=72) as executor:
        # 提交所有CSV文件的处理任务
        future_to_csv = {
            executor.submit(process_csv, os.path.join(csv_dir, csv_file), source, csv_file.replace(".csv", '')): csv_file
            for csv_file in csv_list
        }
        
        # 使用tqdm显示进度
        for future in tqdm.tqdm(as_completed(future_to_csv), total=len(csv_list), desc="Processing CSV files"):
            csv_file = future_to_csv[future]
            try:
                subject_dict = future.result()
                merge_subject_dicts(final_subject_dict, subject_dict)
            except Exception as exc:
                print(f"{csv_file} generated an exception: {exc}")
    
    print(f"Total unique subject_ids: {len(final_subject_dict)}")
    
    # 保存所有数据到jsonl文件
    save_item_whole(final_subject_dict, source, output_root_dir)

if __name__ == "__main__":
    main()
