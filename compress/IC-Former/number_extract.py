import re

def extract_numbers(text):
    """提取文本中的数字，并记录值、类型和位置"""
    patterns = {
        'date': r'\d{4}-\d{2}-\d{2}',           # 日期：xxxx-xx-xx
        'time': r'\d{2}:\d{2}:\d{2}',           # 时间：xx:xx:xx
        'decimal': r'-?\d+\.\d+',               # 小数：-xxx.xxx
        'integer': r'-?\b\d+\b(?![\d\.:-])'     # 整数：-xxx（避免匹配日期/时间/小数）
    }
    numbers = []
    for type_name, pattern in patterns.items():
        matches = re.finditer(pattern, text)
        for match in matches:
            numbers.append({
                'value': match.group(),
                'type': type_name,
                'start': match.start(),
                'end': match.end()
            })
    # 按位置排序，避免替换时位置错乱
    numbers.sort(key=lambda x: x['start'])
    return numbers

def replace_numbers_with_placeholder(text, numbers):
    """将数字替换为占位符"""
    placeholder = '<NUMBER>'
    offset = 0
    for num in numbers:
        start = num['start'] + offset
        end = num['end'] + offset
        text = text[:start] + placeholder + text[end:]
        offset += len(placeholder) - (end - start)
    return text, numbers  # 返回替换后的文本和数字信息
