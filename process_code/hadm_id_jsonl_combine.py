import os
import json
import jsonlines
from datetime import datetime, timedelta
from collections import defaultdict
from dateutil import parser
import tqdm
import pandas as pd
from joblib import Parallel, delayed

PATIENTS_SORTED_DIR = "/Users/tangbohao/Desktop/mimic/test"
subjects = os.listdir(PATIENTS_SORTED_DIR)

# 所有可能的时间字段，包括缺少具体时间的日期字段
TIME_FIELDS = {
    'admittime','charttime','deathtime','dischtime','edouttime','edregtime',
    'entertime','intime','ordertime','outtime','starttime','stoptime','storetime',
    'transfertime','verifiedtime','storedate','chartdate'
}

def read_jsonl(jsonl_path):
    """读取jsonl文件"""
    data = []
    with jsonlines.open(jsonl_path, 'r') as reader:
        for line in reader:
            data.append(line)
    return data

def write_jsonl(data, jsonl_path):
    """写入jsonl文件"""
    with jsonlines.open(jsonl_path, 'w') as writer:
        for d in data:
            writer.write(d)

def parse_time(timestr):
    """
    优先尝试解析完整的日期时间格式，如果失败，则尝试仅日期格式。
    如果输入不是字符串或解析失败，返回None。
    """
    if not isinstance(timestr, str):
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(timestr, fmt)
        except ValueError:
            continue
    return None

def get_hadm_time_range(admission_data):
    """
    从有 hadm_id 的记录中提取 admittime 和 dischtime，返回 (admit_dt, discharge_dt)。
    如果缺少时间字段，则赋予默认值。
    """
    admit_dt = datetime.min
    discharge_dt = datetime.max
    for item in admission_data:
        if item.get('file_name') == 'admissions':
            adt = item.get('admittime')
            dst = item.get('dischtime')
            admit_dt = parse_time(adt) if adt else None
            discharge_dt = parse_time(dst) if dst else None
            # 只从admissions中获取一次即可
            break
    # 若缺少时间，设置默认值
    if not admit_dt:
        admit_dt = datetime.min
    if not discharge_dt:
        # 如果缺少出院时间，设置为未来的一个时间点
        discharge_dt = datetime.max
    return admit_dt, discharge_dt

def extract_time_range_no_hadm(item):
    """
    对没有hadm_id的记录，尝试从所有时间字段获取最早与最晚时间，
    若完全没有时间则返回(None, None)。
    """
    times = []
    for field in TIME_FIELDS:
        if field in item and item[field]:
            t = parse_time(item[field])
            if t:
                times.append(t)
    if not times:
        return None, None
    return min(times), max(times)

def extract_discharge_summaries(admission_data):
    """从 note_type 中包含 'ds' 的记录里提取文本作为出院总结。"""
    summaries = []
    for item in admission_data:
        if item.get('file_name') == 'note':
            note_type = item.get('note_type', '').lower()
            if 'ds' in note_type:
                summaries.append(item.get('text', ''))
    return "\n".join(summaries)


def process_subject_by_hadm(subject_dir, output_dir=None):
    """
    将 subject 下的 combined.jsonl 按 hadm_id 切分，并在当前住院前插入过去的出院小结历史。
    同时将无 hadm_id 的数据按照时间范围插入对应hadm_id的记录中。

    最终在 subject_id 下的 "hadms" 子目录中，每个hadm_id会有一个 hadm_id.jsonl 文件。
    """
    if output_dir is None:
        output_dir = subject_dir
    
    combined_path = os.path.join(subject_dir, 'combined.jsonl')
    if not os.path.exists(combined_path):
        print(f"No combined.jsonl found in {subject_dir}")
        return
    
    data = read_jsonl(combined_path)
    patient_items = [d for d in data if d.get('file_name') == 'patients']  # 找到 file_name="patients" 的记录，后面要复制到所有 hadm_id 中
    # 按 hadm_id 分组
    hadm_groups = defaultdict(list)  
    no_hadm_data = []  # 存储无 hadm_id 数据的列表
    for item in data:
        if item.get('file_name') == "patients":
            continue
        if 'hadm_id' in item and item['hadm_id'] is not None:
            hid = item['hadm_id']
            # 检查 hadm_id 是否为 NaN
            if not (isinstance(hid, float) and json.dumps(hid) == 'NaN'):
                hadm_groups[hid].append(item)
            else:
                # 如果 hadm_id 存在但为 NaN，不进行任何操作
                no_hadm_data.append(item)
        else:
            # 如果没有 'hadm_id' 键，添加 'hadm_id' 并设置为 NaN
            item['hadm_id'] = float('nan')  # 在 JSON 中将被序列化为 null
            no_hadm_data.append(item)
    
    # 获取每个 hadm_id 的住院时间区间
    hadm_ranges = {}
    for hid, items in hadm_groups.items():
        admit_dt, discharge_dt = get_hadm_time_range(items)
        hadm_ranges[hid] = (admit_dt, discharge_dt)
    
    # 根据入院时间排序 hadm_id
    sorted_hadm_ids = sorted(hadm_groups.keys(), key=lambda x: hadm_ranges[x][0])
    
    # 将无 hadm_id 的数据分配到最接近的 hadm_id
    for nh_item in no_hadm_data:
        start, _ = extract_time_range_no_hadm(nh_item)
        if start:
            # 初始化最小时间差
            min_time_diff = timedelta.max
            closest_hadm_id = None
            for hid in sorted_hadm_ids:
                adm_dt, dis_dt = hadm_ranges[hid]
                if adm_dt <= start <= dis_dt:
                    # 时间在区间内，直接分配
                    closest_hadm_id = hid
                    break
                else:
                    # 计算与入院和出院时间的距离，取最近的时间点
                    time_diff = min(abs((start - adm_dt)), abs((start - dis_dt)))
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        closest_hadm_id = hid
            if closest_hadm_id:
                new_item = dict(nh_item)
                new_item['hadm_id'] = closest_hadm_id
                hadm_groups[closest_hadm_id].append(new_item)
        else:
            continue  # 无法解析时间，跳过

    
    hadms_output_dir = os.path.join(output_dir, "hadms")
    os.makedirs(hadms_output_dir, exist_ok=True)

    # 仅记录上一次住院的出院小结
    last_discharge_summary = None
    for idx, hid in enumerate(sorted_hadm_ids):
        admission_data = hadm_groups[hid]
        # 按最早时间排序
        admission_data.sort(key=lambda x: (extract_time_range_no_hadm(x)[0] or datetime.min))
        current_summary = extract_discharge_summaries(admission_data)

        if last_discharge_summary and admission_data:
            insertion_time = (
                admission_data[0].get('charttime') or
                admission_data[0].get('chartdate') or
                hadm_ranges[hid][0].strftime('%Y-%m-%d %H:%M:%S')
            )
            history_item = {
                'file_name': 'note',
                'hadm_id': hid,
                'subject_id': admission_data[0].get('subject_id'),
                'note_type': 'History from last admission',
                'charttime': insertion_time,
                'text': "Past Admission History:\n" + last_discharge_summary
            }
            admission_data.insert(0, history_item)
        if current_summary:
            last_discharge_summary = current_summary

        # 将 'patients' 信息插入到 admission_data 中
        for p_item in patient_items:
            new_p_item = dict(p_item)
            new_p_item['hadm_id'] = hid
            admission_data.append(new_p_item)

        # 按 file_name 分组
        organized_data = []
        cur_name = None
        bundle = None
        for i, row in enumerate(admission_data):
            fname = row.get('file_name', '')
            if fname != cur_name:
                if i != 0 and bundle is not None:
                    organized_data.append(bundle)
                cur_name = fname
                bundle = {"file_name": fname, "hadm_id": hid, "items": [row]}
            else:
                bundle["items"].append(row)
        if bundle is not None:
            organized_data.append(bundle)

        # 写入对应的 hadm_id 文件
        out_path = os.path.join(hadms_output_dir, f"{hid}.jsonl")
        write_jsonl(organized_data, out_path)

def process_subject(subject):
    """并行处理每个subject"""
    process_subject_by_hadm(PATIENTS_SORTED_DIR + '/' + subject )


Parallel(n_jobs=-1)(delayed(process_subject)(subject) for subject in tqdm.tqdm(subjects))
