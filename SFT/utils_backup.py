import os 
import pandas as pd
import json
import datetime
import jsonlines
from torch.utils.data import Dataset, DataLoader
import tqdm
from functools import *
import random
import copy

SPLIT_LINE = "\n\n-------------------------------------------------------------------------------------\n\n"

def read_jsonl(jsonl_dir):
    data_list = []
    with open(jsonl_dir, 'r') as file:
        for line in file:
            data_list.append(json.loads(line))
    return data_list

def process_icd(index_csv):
    inflect_index = {"9":{},"10":{}}
    with open(index_csv) as f:
        icd_index = pd.read_csv(f)
    for index in tqdm.tqdm(range(len(icd_index))):
        sample = icd_index.iloc[index]
        inflect_index[str(sample["icd_version"])][str(sample["icd_code"])] = sample["long_title"]
    return inflect_index
    
def process_hcpcs_item(index_csv):
    inflect_index = {}
    prompt ="Hcpcs_event_name: {short_description} \n"
    with open(index_csv) as f:
        index_pd = pd.read_csv(f)
    for index in tqdm.tqdm(range(len(index_pd))):
        sample = index_pd.iloc[index]
        inflect_index[str(sample["code"])] = prompt.format(short_description = sample["short_description"])
    return inflect_index
    
def process_hosp_item(index_csv):
    inflect_index = {}
    #prompt ="Event_name: {label} \nFluid: {fluid} \nCategory: {category} \n"
    prompt = "{label}"
    with open(index_csv) as f:
        index_pd = pd.read_csv(f)
    for index in tqdm.tqdm(range(len(index_pd))):
        sample = index_pd.iloc[index]
        #inflect_index[str(sample["itemid"])] = prompt.format(label = sample["label"], fluid = sample["fluid"], category = sample["category"])
        inflect_index[str(sample["itemid"])] = prompt.format(label = sample["label"])
    return inflect_index
            
    
def process_icu_item(index_csv):
    """
    label,abbreviation,linksto,category,unitname,param_type,lownormalvalue,highnormalvalue
    """
    inflect_index = {}
        
    #prompt ="Event_name: {label} \nAbbreviation: {abbreviation} \nLinksto: {linksto} \nCategory: {category} \nUnitname: {unitname} \nParam_type: {param_type} \nLownormalvalue: {lownormalvalue} \nHighnormalvalue: {highnormalvalue} \n"
    prompt = "{label}"
    with open(index_csv) as f:
        index_pd = pd.read_csv(f)
    for index in tqdm.tqdm(range(len(index_pd))):
        sample = index_pd.iloc[index]
        #inflect_index[str(sample["itemid"])] = prompt.format(label = sample["label"], abbreviation = sample["abbreviation"], linksto = sample["linksto"], category = sample["category"], unitname = sample["unitname"], param_type = sample["param_type"], lownormalvalue = sample["lownormalvalue"], highnormalvalue = sample["highnormalvalue"])
        inflect_index[str(sample["itemid"])] = prompt.format(label = sample["label"])
    return inflect_index

def safe_read(json_element):
    if isinstance(json_element, float) or isinstance(json_element, int):
        json_element = str(json_element)
    
    if isinstance(json_element, str):
        if json_element == 'NaN' or json_element == "nan":
            return ""
    
    if pd.isna(json_element):
        json_element = ""
    
    if json_element is None:
        return ""
    
    return json_element
    
def patients_item_to_free_text(item):
    item_list = item["items"]
    assert len(item_list)==1, print(item_list)
    gender = item_list[0]["gender"]
    age = item_list[0]["anchor_age"]
    
    prompt = "Event_type: Patient, Time: Background\n\nThe patient is a {age}-year-old {gender}."
    return prompt.format(age = age, gender = gender)
    
def labevents_item_to_free_text(item, item_indexing):
    item_list = item["items"]
    start_time = item_list[0]["charttime"]
    end_time = item_list[-1]["charttime"]
    prompt = "Event_type: Labevent, Time: {start_time} -> {end_time}\n\n".format(start_time=start_time, end_time=end_time)
    
    chart_item = []
    chart_prompt = "{item_name}: {valuenum} {valueuom}  ({ref_range_lower}, {ref_range_upper})--------{flag} ({comments})"
    for item in item_list:
        item_name = item_indexing[safe_read(item["itemid"])]
        valuenum = safe_read(item["valuenum"])
        valueuom = safe_read(item["valueuom"]) 
        ref_range_lower = safe_read(item["ref_range_lower"]) 
        ref_range_upper = safe_read(item["ref_range_upper"]) 
        flag = safe_read(item["flag"]) 
        comments = safe_read(item["comments"]) 
        flag = "normal" if flag =="" else flag
        if valuenum != "":
            chart_item.append(chart_prompt.format(item_name=item_name,valuenum=valuenum, valueuom= valueuom, ref_range_lower = ref_range_lower, ref_range_upper=ref_range_upper,flag=flag, comments = comments))
    
    chart_item_str = "\n".join(chart_item)
    
    prompt = prompt + chart_item_str
    
    return prompt


def omr_item_to_free_text(item):
    item_list = item["items"]
    start_time = item_list[0]["chartdate"]
    end_time = item_list[-1]["chartdate"]
    prompt = "Event_type: OMR, Time: {start_time} -> {end_time}\n\n".format(start_time=start_time, end_time=end_time)
    
    chart_item = []
    chart_prompt = "{result_name}: {result_value}"
    for item in item_list:
        result_name = safe_read(item["result_name"])
        result_value = safe_read(item["result_value"])
        chart_item.append(chart_prompt.format(result_name=result_name,result_value=result_value))
    chart_item_str = "\n".join(chart_item)
    prompt = prompt + chart_item_str
    
    return prompt

def transfers_item_to_free_text(item):
    item_lists = item["items"]
    chart_strs_list = []
    for item_list in item_lists:
    # assert len(item_list)==1, print(item_list)
        start_time = safe_read(item_list["intime"])
        end_time = safe_read(item_list["outtime"])
        eventtype = safe_read(item_list["eventtype"])
        careunit = safe_read(item_list["careunit"])
        transfer_type = eventtype + '/' + careunit
        prompt = "Event_type: Transfers ({transfer_type}), Time: {start_time} -> {end_time}\n\n".format(transfer_type = transfer_type, start_time=start_time, end_time=end_time)
        chart_strs_list.append(prompt)
    chart_strs = SPLIT_LINE.join(chart_strs_list)
    # chart_item = []
    # chart_prompt = "{result_name}: {result_value}"
    # for item in item_list:
    #     result_name = safe_read(item["result_name"])
    #     result_value = safe_read(item["result_value"])
    #     chart_item.append(chart_prompt.format(result_name=result_name,result_value=result_value))
    # chart_item_str = "\n".join(chart_item)
    # prompt = prompt + chart_item_str
    
    return chart_strs

def poe_item_to_free_text(item):
    item_list = item["items"]
    start_time = safe_read(item_list[0]["ordertime"])
    end_time = safe_read(item_list[-1]["ordertime"])

    prompt = "Event_type: POE, Time: {start_time} -> {end_time}\n\n".format(start_time=start_time, end_time=end_time)
    
    chart_item = []
    chart_prompt = "{order_type}: {order_subtype}"
    
    for item in item_list:
        order_type = safe_read(item["order_type"])
        order_subtype = safe_read(item["order_subtype"])
        chart_item.append(chart_prompt.format(order_type=order_type,order_subtype=order_subtype))
    chart_item_str = "\n".join(chart_item)
    prompt = prompt + chart_item_str
    
    return prompt

def services_item_to_free_text(item):
    item_list = item["items"]
    start_time = safe_read(item_list[0]["transfertime"])
    end_time = safe_read(item_list[-1]["transfertime"])

    prompt = "Event_type: Service, Time: {start_time} -> {end_time}\n\n".format(start_time=start_time, end_time=end_time)
    
    chart_item = []
    chart_prompt = "{curr_service}"
    
    for item in item_list:
        curr_service = item["curr_service"]
        chart_item.append(chart_prompt.format(curr_service=curr_service))
    chart_item_str = "\n".join(chart_item)
    prompt = prompt + chart_item_str
    
    return prompt

def note_item_to_free_text(item):
    item_list = item["items"]
    start_time = safe_read(item_list[0]["charttime"])
    end_time = safe_read(item_list[-1]["charttime"])

    prompt = "Event_type: Note, Time: {start_time} -> {end_time}\n\n".format(start_time=start_time, end_time=end_time)
    
    chart_item = []
    chart_prompt = "Note Type: {note_type} \nText: \n{text}"
    
    for item in item_list:
        note_type = safe_read(item["note_type"])
        text = safe_read(item["text"])
        chart_item.append(chart_prompt.format(note_type=note_type, text= text))
    chart_item_str = "\n".join(chart_item)
    prompt = prompt + chart_item_str
    
    return prompt
    
def admissions_item_to_free_text(item):
    item_list = item["items"]
    start_time = safe_read(item_list[0]["admittime"])
    end_time = safe_read(item_list[-1]["dischtime"])
    
    # "admission_location": "EMERGENCY ROOM", "discharge_location"
    admission_location = safe_read(item_list[0]["admission_location"])
    discharge_location = safe_read(item_list[0]["discharge_location"])
    prompt = "Event_type: Admission({admission_location} -> {discharge_location}), Time: {start_time} -> {end_time}\n\n".format(admission_location= admission_location, discharge_location= discharge_location, start_time=start_time, end_time=end_time)
    
    # chart_item = []
    # chart_prompt = "Note Type: {note_type} \n Text: {text}"
    
    # for item in item_list:
    #     note_type = safe_read(item["note_type"])
    #     text = safe_read(item["text"])
    #     chart_item.append(chart_prompt.format(note_type=note_type, text= text))
    # chart_item_str = "\n".join(chart_item)
    # prompt = prompt + chart_item_str
    
    return prompt

def emar_item_to_free_text(item):
    item_list = item["items"]
    start_time = safe_read(item_list[0]["charttime"])
    end_time = safe_read(item_list[-1]["charttime"])
    prompt = "Event_type: Medication, Time: {start_time} -> {end_time}\n\n".format(start_time=start_time, end_time=end_time)
    
    chart_item = []
    chart_prompt = "{medication} "
    
    for item in item_list:
        medication = safe_read(item["medication"])
        chart_item.append(chart_prompt.format(medication=medication))
    chart_item_str = ", ".join(chart_item)
    prompt = prompt + chart_item_str
    
    return prompt

def microbiologyevents_item_to_free_text(item):
    """
    {"microevent_id": 20, "subject_id": 10000032, "hadm_id": NaN, "micro_specimen_id": 5842819, "order_provider_id": NaN, "chartdate": "2180-06-26 00:00:00", 
    "charttime": "2180-06-26 18:30:00", "spec_itemid": 70079, "spec_type_desc": "URINE", "test_seq": 1, "storedate": "2180-06-29 00:00:00", "storetime": "2180-06-29 14:32:00", 
    "test_itemid": 90039, "test_name": "URINE CULTURE", "org_itemid": 80053.0, "org_name": "ENTEROCOCCUS SP.", "isolate_num": 1.0, "quantity": NaN, "ab_itemid": 90004.0, 
    "ab_name": "AMPICILLIN", "dilution_text": "<=2", "dilution_comparison": "<=        ", "dilution_value": 2.0, 
    "interpretation": "S", "comments": "MIXED BACTERIAL FLORA ( >= 3 COLONY TYPES), CONSISTENT WITH SKIN AND/OR GENITAL CONTAMINATION.  ", "file_name": "microbiologyevents"}
    """
    item_list = item["items"]
    start_time = item_list[0]["charttime"]
    end_time = item_list[-1]["charttime"]
    prompt = "Event_type: Labevent, Time: {start_time} -> {end_time}\n\n".format(start_time=start_time, end_time=end_time)
    
    
    chart_item = []
    chart_prompt = "{test_name}({ab_name}): {dilution_text}--------{interpretation} ({comments})"
    for item in item_list:
        test_name = safe_read(item["test_name"])
        dilution_text = safe_read(item["dilution_text"])
        interpretation = safe_read(item["interpretation"]) 
        comments = safe_read(item["comments"]) 
        ab_name = safe_read(item["ab_name"])
        
        chart_item.append(chart_prompt.format(test_name=test_name,dilution_text=dilution_text, interpretation= interpretation, comments = comments, ab_name=ab_name))
    
    chart_item_str = "\n".join(chart_item)
    
    prompt = prompt + chart_item_str
    
    return prompt


def diagnoses_icd_item_to_free_text(item, item_indexing):
    
    item_list = item["items"]
    #start_time = item_list[0]["charttime"]
    #end_time = item_list[-1]["charttime"]
    prompt = "Event_type: Diagnoses, Time: Background\n\n"
    
    
    chart_item = []
    chart_prompt = "{disease_name}"
    for item in item_list:
        icd_code = safe_read(item["icd_code"])
        icd_version = safe_read(item["icd_version"])
        disease_name = item_indexing[icd_version][icd_code]
        
        chart_item.append(chart_prompt.format(disease_name=disease_name))
    
    chart_item_str = ", ".join(chart_item)
    
    prompt = prompt + chart_item_str
    
    return prompt
    
def procedures_icd_item_to_free_text(item, item_indexing):

    item_list = item["items"]
    start_time = item_list[0]["chartdate"]
    end_time = item_list[-1]["chartdate"]
    prompt = "Event_type: Procedures, Time: {start_time} -> {end_time}\n\n".format(start_time=start_time, end_time=end_time)
    
    
    chart_item = []
    chart_prompt = "{procedure_name}"
    for item in item_list:
        icd_code = safe_read(item["icd_code"])
        icd_version = safe_read(item["icd_version"])
        procedure_name = item_indexing[icd_version][icd_code]
        
        chart_item.append(chart_prompt.format(procedure_name=procedure_name))
    
    chart_item_str = ", ".join(chart_item)
    
    prompt = prompt + chart_item_str
    
    return prompt

def ed_item_to_free_text(item):
    item_lists = item["items"]
    #assert len(item_list)==1, print(item_list)
    chart_items = []
    for item_list in item_lists:
        start_time = item_list["intime"]
        end_time = item_list["outtime"]
        prompt = "Event_type: ED, Time: {start_time} -> {end_time}\n\n".format(start_time=start_time, end_time=end_time)
        
        chart_item = []
        for item in item_list["sub_items"]:
            # item_name = item_indexing[safe_read(item["itemid"])]
            
            # chart_prompt.format(item_name = item_name)
            
            file_name = item["file_name"]
            if file_name == "triage":
                temperature = safe_read(item["temperature"])
                heartrate = safe_read(item["heartrate"])
                resprate = safe_read(item["resprate"])
                o2sat = safe_read(item["o2sat"])
                sbp = safe_read(item["sbp"])
                dbp = safe_read(item["dbp"])
                pain = safe_read(item["pain"])
                acuity = safe_read(item["acuity"])
                chiefcomplaint = safe_read(item["chiefcomplaint"])
                
                chart_prompt = "Triage: temperature: {temperature}, heartrate: {heartrate}, resprate: {resprate}, o2sat: {o2sat}, sbp: {sbp}, dbp: {dbp}, pain: {pain}, acuity: {acuity}, chiefcomplaint: {chiefcomplaint}".format(temperature = temperature, heartrate = heartrate, resprate = resprate, o2sat = o2sat, sbp=sbp, dbp=dbp, pain= pain, acuity=acuity, chiefcomplaint= chiefcomplaint)
                chart_item.append(chart_prompt)
            
            if file_name == "diagnosis":
                # icd_code = safe_read(item["icd_code"])
                # icd_version = safe_read(item["icd_version"])
                
                # print(icd_version, icd_code)
                disease_name = safe_read(item["icd_title"]).lower().capitalize()
                
                chart_prompt = "Diagnosis: {disease_name}".format(disease_name = disease_name)
                chart_item.append(chart_prompt)
            
            if file_name == "medrecon":
                name = safe_read(item["name"])
                
                chart_prompt = "Medrecon: {name}".format(name = name)
                chart_item.append(chart_prompt)
            
            if file_name == "pyxis":
                name = safe_read(item["name"])
                
                chart_prompt = "Pyxis: {name}".format(name = name)
                chart_item.append(chart_prompt)
            
            if file_name == "vitalsign":
                temperature = safe_read(item["temperature"])
                heartrate = safe_read(item["heartrate"])
                resprate = safe_read(item["resprate"])
                o2sat = safe_read(item["o2sat"])
                sbp = safe_read(item["sbp"])
                dbp = safe_read(item["dbp"])
                pain = safe_read(item["pain"])
                rhythm = safe_read(item["rhythm"])
                
                chart_prompt = "Vitalsign: temperature: {temperature}, heartrate: {heartrate}, resprate: {resprate}, o2sat: {o2sat}, sbp: {sbp}, dbp: {dbp}, pain: {pain}, rhythm: {rhythm}".format(temperature = temperature, heartrate = heartrate, resprate = resprate, o2sat = o2sat, sbp=sbp, dbp=dbp, pain= pain, rhythm= rhythm)
                chart_item.append(chart_prompt)

        chart_item_str = " \n".join(chart_item)
        
        prompt = prompt + chart_item_str
        chart_items.append(prompt)
        
    chart_items_str = SPLIT_LINE.join(chart_items)
    return chart_items_str

def icu_item_to_free_text(item, item_indexing):
    item_lists = item["items"]
    chart_items = []
    for item_list in item_lists:
        start_time = item_list["intime"]
        end_time = item_list["outtime"]
        prompt = "Event_type: ICU, Time: {start_time} -> {end_time}\n\n".format(start_time=start_time, end_time=end_time)
        
        chart_item = []
        
        for item in item_list["sub_items"]:
            #print(item)
            item_name = item_indexing[safe_read(item["itemid"])]
            chart_prompt = "{item_name}: "
            chart_prompt.format(item_name = item_name)
            
            file_name = item["file_name"]

            # icu里面的图表项目
            if file_name == "chartevents":
                value = safe_read(item["value"])
                valuenum = safe_read(item["valuenum"])
                valueuom = safe_read(item["valueuom"])
                warning = safe_read(item["warning"])
                chart_prompt = "Chartevents " + chart_prompt + "{value}, {valuenum} {valueuom}, warning: {warning}".format(value = value, valuenum = valuenum, valueuom = valueuom, warning = warning)
                chart_item.append(chart_prompt)
            
            if file_name == "inputevent":
                amount = safe_read(item["amount"])
                amountuom = safe_read(item["amountuom"])
                rate = safe_read(item["rate"])
                rateuom = safe_read(item["rateuom"])
                
                chart_prompt = "Inputevent " + chart_prompt + "{amount}, {amountuom} {valueuom}, rate: {rate}, {rateuom}".format(amount = amount, amountuom = amountuom, rate = rate, rateuom = rateuom)
                chart_item.append(chart_prompt)
            
            if file_name == "output":
                value = safe_read(item["value"])
                valueuom = safe_read(item["valueuom"])
                
                chart_prompt = "Output " + chart_prompt + "{value} {valueuom}".format(value = value, valueuom = valueuom, rate = rate, rateuom = rateuom)
                chart_item.append(chart_prompt)
            
            # 新增 icustays 处理
            if file_name == "icustays":
                fc = safe_read(item.get("first_careunit"))
                lc = safe_read(item.get("last_careunit"))
                ls = safe_read(item.get("los"))
                chart_prompt = "Icustays " + chart_prompt + f"FirstCareUnit: {fc}, LastCareUnit: {lc}, LOS: {ls}"
                chart_item.append(chart_prompt)

            # 新增 ingredientevents 处理
            if file_name == "ingredientevents":
                amount = safe_read(item.get("amount"))
                amountuom = safe_read(item.get("amountuom"))
                rate = safe_read(item.get("rate"))
                rateuom = safe_read(item.get("rateuom"))
                chart_prompt = "Ingredientevents " + chart_prompt + "{amount} {amountuom}, rate: {rate} {rateuom}".format(amount = amount, amountuom = amountuom, rate = rate, rateuom = rateuom)
                chart_item.append(chart_prompt)

            # 新增 procedureevents 处理
            if file_name == "procedureevents":
                value = safe_read(item.get("value"))
                valueuom = safe_read(item.get("valueuom"))
                status = safe_read(item.get("statusdescription"))
                chart_prompt = "Procedureevents " + chart_prompt + "{value} {valueuom}, status: {status}".format(value = value, valueuom = valueuom, status = status)
                chart_item.append(chart_prompt)
    
        chart_item_str = prompt + " \n".join(chart_item)
        
        chart_items.append(chart_item_str)
    
    chart_items_str = SPLIT_LINE.join(chart_items) 
    return chart_items_str

def prescriptions_item_to_free_text(item):

    item_lists = item["items"]  # List of prescription entries
    chart_strs_list = []        # Will store the readable text for each prescription

    for single_prescription in item_lists:
        start_time = safe_read(single_prescription["starttime"])
        end_time = safe_read(single_prescription["stoptime"])
        drug_name = safe_read(single_prescription["drug"])
        dose_val = safe_read(single_prescription["dose_val_rx"])
        dose_unit = safe_read(single_prescription["dose_unit_rx"])
        # route = safe_read(single_prescription["route"])
        # freq = safe_read(single_prescription["doses_per_24_hrs"])  # Number of doses per 24 hours

        prompt = (
            f"Event_type: Prescriptions, Time: {start_time} -> {end_time}\n\n"
            f"Drug: {drug_name}\n"
            f"Dose: {dose_val}{dose_unit}\n"
            # f"Route: {route}\n"
            # f"Frequency: {freq} times per 24 hours"
        )

        chart_strs_list.append(prompt)

    # Join all prescription descriptions with the predefined SPLIT_LINE separator
    chart_strs = SPLIT_LINE.join(chart_strs_list)
    return chart_strs



def similarity_sample(target_items_id, choice_list):
    '''
    相似项采样：
        从CSV文件加载候选项列表
        为每个目标项找到相似的候选项
        从候选集合中随机采样
        将采样结果与目标项合并并打乱顺序
    '''
    # print(choice_list)
    # 如果choice_list是文件路径，则从CSV文件中加载。
    if os.path.exists(choice_list):
        choice_list = pd.read_csv(choice_list)
        
    
    candidates_list = []
    target_items = []
    # print(target_items_id)
    for item in target_items_id:
        # print(item)
        row = choice_list[choice_list['item'] == item]
        if row['item'].values.flatten().tolist() == []:
            row = choice_list[choice_list['item_id'] == item]
        if row['item'].values.flatten().tolist() == []:
            continue
        candidates_list.extend(row[[f"top_{i}" for i in range(10, 50)]].values.flatten().tolist())
        # print(row['item'].values.flatten().tolist())
        target_items.append(row['item'].values.flatten().tolist()[0]) # id2str

    # 转换为 set
    candidates_list = set(candidates_list)
    # 从 set 中采样，例如采样为target_items的三倍
    sample_size = 3 * len(target_items)
    if sample_size > len(candidates_list):
        sampled_items = list(candidates_list)
    else:
        sampled_items = random.sample(candidates_list, sample_size)
    
    candidate_items = sampled_items + target_items
    random.shuffle(candidate_items)
    
    return candidate_items, target_items

# def filter_24h_data(self, patient_trajectory_list):
#     """
#     过滤 patient_trajectory_list，只保留入院24小时内的数据。
#     """
#     TIME_FIELDS = {
#         'admittime','charttime','deathtime','dischtime','edouttime','edregtime',
#         'entertime','intime','ordertime','outtime','starttime','stoptime','storetime',
#         'transfertime','verifiedtime','storedate','chartdate'
#     }

#     admission_time = None
#     # 在患者轨迹中查找 admissions 项，获取入院时间
#     for item in patient_trajectory_list:
#         if item.get("file_name","") == "admissions":
#             try:
#                 admission_str = item["items"][0].get("admittime","")
#                 admission_time = pd.to_datetime(admission_str)
#                 break
#             except:
#                 pass
    
#     if not admission_time:
#         # 若无法获取入院时间，默认不过滤
#         return patient_trajectory_list
    
#     filtered_trajectory = []
#     cutoff_time = admission_time + pd.Timedelta(hours=24)
    
#     for item in patient_trajectory_list:
#         item_list = item.get("items", [])
#         # 筛选 item_list 中 charttime 在 24 小时内的
#         valid_events = []
#         for evt in item_list:
#             earliest_time = None
#             for f in TIME_FIELDS:
#                 t_str = evt.get(f, "")
#                 try:
#                     t_dt = pd.to_datetime(t_str)
#                     if earliest_time is None or t_dt < earliest_time:
#                         earliest_time = t_dt
#                 except:
#                     pass
#             if earliest_time is None or earliest_time <= cutoff_time:
#                 valid_events.append(evt)
        
#         # 若筛选后仍有数据，则加入
#         if len(valid_events) > 0:
#             new_item = dict(item)
#             new_item["items"] = valid_events
#             filtered_trajectory.append(new_item)
    
#     return filtered_trajectory