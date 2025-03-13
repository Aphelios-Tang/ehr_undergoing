import sys

def count_lines(file_path):
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for count, line in enumerate(f, 1):
            pass
    return count

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python count_jsonl_lines.py <jsonl文件路径>")
        sys.exit(1)
    file_path = sys.argv[1]
    print("文件行数：", count_lines(file_path))