import pandas as pd
import json

def filter_24h_data(patient_trajectory_list):
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
                print(f"无法解析 admittime: {e}")
                pass

    if not admission_time:
        # 若无法获取入院时间，默认不过滤
        return patient_trajectory_list

    print("admission_time:", admission_time)
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

# 从本地读取数据
with open("/Users/tangbohao/Desktop/mimic/test/12007761/hadms/29615498.jsonl", "r") as f:
    # patient_trajectory_list = [json.loads(line.replace('NaN', 'null')) for line in f]
    patient_trajectory_list = [json.loads(line) for line in f]

# 过滤数据
filtered_trajectory = filter_24h_data(patient_trajectory_list)
print(filtered_trajectory)