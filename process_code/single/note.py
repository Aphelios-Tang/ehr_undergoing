import os 
import pandas as pd
import json
import datetime
import jsonlines

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            print("MyEncoder-datetime.datetime")
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        if isinstance(obj, int):
            return int(obj)
        if isinstance(obj, float):
            return float(obj)
        #elif isinstance(obj, array):
        #    return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
        
def save_item(subject_id, source, item, root_dir = '/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/my_MIMIC_IV/patients/'):
    patient_dir = root_dir + str(subject_id)
    
    if not os.path.exists(patient_dir):
        try:
            os.mkdir(patient_dir)
        except:
            pass
    
    if os.path.exists(patient_dir+'/' +source+'.jsonl'):
        with jsonlines.open(patient_dir+'/' +source+'.jsonl',  'a') as f:
            f.write(item)
    else:
        with jsonlines.open(patient_dir+'/' +source+'.jsonl',  'w') as f:
            f.write(item)


def save_item_whole(source, item, index, root_dir = '/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/my_MIMIC_IV/patients/'):
    
    for subject_id in tqdm.tqdm(list(item.keys())):
        patient_dir = root_dir + str(subject_id)

        if not os.path.exists(patient_dir):
            try:
                os.mkdir(patient_dir)
            except:
                pass
    
        with jsonlines.open(patient_dir+'/' +source+'.jsonl',  'w') as f:
            for ii in item[subject_id]:
                f.write(ii)
                
subject_dict = {}   
def work(csv_path, source, file_name, index_csv):
    
    with open(csv_path,'r') as f:
        items = pd.read_csv(f)
    print(len(items))
    print(index_csv)
    # print(list(items.keys()))
    # input()
    if "subject_id" in list(items.keys()):
        print(file_name)
        for index in tqdm.tqdm(range(len(items))):
            sample = items.iloc[index].to_dict()
            # print(sample)
            # input()
            # discharge_note = {}
            #sample['file_name'] = file_name
            # for key in list(sample.keys()):
            #     discharge_note[key] = sample[key]
            subject_id = sample["subject_id"]
            try:
                subject_dict[subject_id].append(sample)
            except:
                subject_dict[subject_id] = [sample]

        #save_item_whole(source=source, index = index_csv, item = subject_dict)
    
        
import tqdm

root_path = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/decompressed_MIMIC_IV"
source = "note"

csv_list = os.listdir(root_path + '/' + source)
index_csv = 0
for csv_file in csv_list:
    if (".csv" not in csv_file) or (csv_file in ["discharge_detail.csv","radiology_detail.csv"]):
        continue
    else:
        work(csv_path = root_path + '/' + source+ '/' + csv_file, source = source, file_name = csv_file.replace(".csv",''), index_csv = index_csv)
        index_csv = index_csv+1

save_item_whole(source=source, index = index_csv, item = subject_dict)