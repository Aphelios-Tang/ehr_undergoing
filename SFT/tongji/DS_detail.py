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

def parse_subsections(section_text):
    # 使用正则分割所有小标题，如 "Admission Examination:"、"VS:"、"HEENT:" 等
    parts = re.split(r'([A-Za-z0-9/\s]+:)', section_text)
    # 例：["", "Admission Examination:", " VS: T 98.5F...", "HEENT:", " Sclerae non-icteric...", ...]

    subsections = {}
    current_subhead = None
    buffer = []

    for part in parts:
        if re.match(r'^[A-Za-z0-9/\s]+:$', part.strip()):
            # 如果已有小标题内容，先保存
            if current_subhead and buffer:
                subsections[current_subhead.strip()] = clean_text("".join(buffer).strip())
                buffer = []
            # 当前分割段是新的小标题
            current_subhead = part
        else:
            # 当前分割段是内容
            buffer.append(part)

    # 收尾
    if current_subhead and buffer:
        subsections[current_subhead.strip()] = clean_text("".join(buffer).strip())

    return subsections

def get_section_dict(text):

    results = {}
    # 先手动提取首段作为“Patients”部分
    first_pattern = r"^(.*?)\n *Chief Complaint:"
    match = re.search(first_pattern, text, re.DOTALL)
    if match:
        first_section = clean_text(match.group(1))
    else:
        first_section = ""

    results = {
        "Patients": {
            "content": first_section,
            "subsections": parse_subsections(first_section)
        }
    }
    
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
        if i < len(sections) - 1:
            next_section = sections[i + 1]
            main_text = extract_section(text, section, next_section)
        else:
            main_text = extract_section(text, section)

        # 返回分级结构
        results[section] = {
            "content": main_text,
            "subsections": parse_subsections(main_text)
        }
    
    return results

if __name__ == "__main__":
    jsonl_dir = "/Users/tangbohao/Files/科研资料/EHR-decision/mimic-code/SFT/tongji/history2.jsonl"
    data_list = read_jsonl(jsonl_dir)
    with open("history2_detail.jsonl", "w") as f:
        for data in data_list:
            text = data["text"]
            section_dict = get_section_dict(text)
            f.write(json.dumps(section_dict, ensure_ascii=False, indent=2) + "\n")

        
