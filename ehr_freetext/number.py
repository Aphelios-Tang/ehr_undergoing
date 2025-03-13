import re
import json

def count_numbers_in_text(text):
    """
    判断文本中数字的个数，并按照规则返回结果
    - 如果没有数字，返回"没有数字"
    - 如果数字个数在1-10之间，返回B
    - 如果数字个数大于10，返回C
    
    规则：
    - 带小数点的数字算作一个数字
    - 日期格式如xxxx-xx-xx算作3个数字
    - 时间格式如xx:xx:xx算作3个数字
    """
    count = 0
    
    # 创建一个处理过的文本副本，避免重复计算
    processed_text = text
    
    # 1. 处理日期格式 (xxxx-xx-xx)
    date_pattern = r'\d{4}-\d{2}-\d{2}'
    dates = []
    for match in re.finditer(date_pattern, processed_text):
        start, end = match.span()
        dates.append((processed_text[start:end], start, end))
    
    count += len(dates) * 3  # 每个日期贡献3个数字
    # print(f"日期: {[d[0] for d in dates]}")
    
    # 从文本中移除已处理的日期（从后往前替换，避免位置变化）
    for date, start, end in sorted(dates, key=lambda x: x[1], reverse=True):
        processed_text = processed_text[:start] + " " * (end - start) + processed_text[end:]
    
    # 2. 处理时间格式 (xx:xx:xx)
    time_pattern = r'\d{2}:\d{2}:\d{2}'
    times = []
    for match in re.finditer(time_pattern, processed_text):
        start, end = match.span()
        times.append((processed_text[start:end], start, end))
    
    count += len(times) * 3  # 每个时间贡献3个数字
    # print(f"时间: {[t[0] for t in times]}")
    
    # 从文本中移除已处理的时间
    for time, start, end in sorted(times, key=lambda x: x[1], reverse=True):
        processed_text = processed_text[:start] + " " * (end - start) + processed_text[end:]
    
    # 3. 处理带小数点的数字
    decimal_pattern = r'\b\d*\.\d+\b'
    decimals = []
    for match in re.finditer(decimal_pattern, processed_text):
        start, end = match.span()
        decimals.append((processed_text[start:end], start, end))
    
    count += len(decimals)  # 每个小数算作1个数字
    # print(f"小数: {[d[0] for d in decimals]}")
    
    # 从文本中移除已处理的小数
    for decimal, start, end in sorted(decimals, key=lambda x: x[1], reverse=True):
        processed_text = processed_text[:start] + " " * (end - start) + processed_text[end:]
    
    # 4. 处理剩余的整数
    integer_pattern = r'\b\d+\b'
    integers = []
    for match in re.finditer(integer_pattern, processed_text):
        start, end = match.span()
        integers.append((processed_text[start:end], start, end))
    
    count += len(integers)
    # print(f"整数: {[i[0] for i in integers]}")
    
    # 根据数字总数返回结果
    if count == 0:
        return count, "A"
    elif 1 <= count <= 16:
        return count, "B"
    elif 17 <= count <= 64:
        return count, "C"
    else:
        return count, "D"



# 执行测试
if __name__ == "__main__":
    json_dir = "/Users/tangbohao/Files/科研资料/EHR-decision/mimic-code/ehr_freetext/reconstruction_item_0.001.jsonl"
    with open(json_dir, 'r') as f:
        file_content = f.readlines()
        i = 0
        for line in file_content:
            i += 1
            if(i>40):
                break   
            data = json.loads(line)
            text = data["text"]
            count, category = count_numbers_in_text(text)
            print(f"数字个数: {count}, 类别: {category}")
        

