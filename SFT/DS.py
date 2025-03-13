import re
import json
import os

def read_jsonl(jsonl_dir):
    data_list = []
    with open(jsonl_dir, 'r') as file:
        for line in file:
            data_list.append(json.loads(line))
    return data_list

def clean_text(text):
    # 将多个空格替换为单个空格，并删除换行符
    return ' '.join(text.replace('\n', ' ').split())

def extract_section(text, section_name, next_section=None):
    if next_section:
        pattern = f"{section_name}:(.*?)\n *{next_section}:"
    else:
        pattern = f"{section_name}:(.*?)\n \nPertinent Results:"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return clean_text(match.group(1).strip())
    return ""

def get_section_dict(text):
    results = {}

    # 先手动提取首段作为“Patients”部分
    first_pattern = r"^(.*?)\n *Chief Complaint:"
    match = re.search(first_pattern, text, re.DOTALL)
    if match:
        first_section = clean_text(match.group(1))
    else:
        first_section = ""

    results["Patients"] = first_section

    sections = [
        "Chief Complaint",
        "Major Surgical or Invasive Procedure",
        "History of Present Illness",
        "Past Medical History",
        "Social History", 
        "Family History",
        "Physical Exam",
        "Pertinent Results",
        "Brief Hospital Course",
        "Medications on Admission",
        "Discharge Medications",
        "Discharge Disposition",
        "Discharge Diagnosis",
        "Discharge Condition",
        "Discharge Instructions"
    ]

    for i, section in enumerate(sections):
        for i, section in enumerate(sections):
        # 如果不是最后一个章节，使用下一个章节作为边界
            if i < len(sections) - 1:
                next_section = sections[i + 1]
                results[section] = extract_section(text, section, next_section)
            else:
                # 最后一个章节使用 "Pertinent Results" 作为边界
                results[section] = extract_section(text, section)
    
    return results

def get_history(text):
    results = get_section_dict(text)
    history_list = ["History of Present Illness", "Past Medical History", "Social History", "Family History"]
    history_section = []
    for section in results:
        if section in history_list:
            history_section.append(section + "\n" + results[section])
    history_text = "\n\n".join(history_section)
    return history_text

def parse_patients_info(first_section):
    """
    从第一段中额外提取“Allergies ”字段。
    """
    # 匹配从“Allergies:”开始，直到下一个可能的字段（如“Attending:”）或文本结尾
    match = re.search(r'Allergies:\s*(.*?)(?=\s+[A-Z][a-zA-Z]+:|$)', first_section, re.DOTALL)
    if match:
        return match.group(1)
    return ""

def get_patients_addention(text):
    results = get_section_dict(text)
    pat = results["Patients"]
    addention = parse_patients_info(pat)
    return "Allergies History: \n" + addention

def discharge_diagnosis(text):
    results = get_section_dict(text)
    return results["Discharge Diagnosis"]


if __name__ == "__main__":

    file = os.path.join("/Users/tangbohao/Desktop/mimic/SFT/tongji/history2.jsonl")
    file_content = read_jsonl(file)
    total = []
    for i, item in enumerate(file_content):
        file_name = item["file_name"]
        if file_name == "note" and item["items"][0]["note_type"] == "DS":
            text = item["items"][0]["text"]
            history_text = get_patients_addention(text)
            print(history_text)
        