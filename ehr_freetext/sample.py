import json
import random
import os

def extract_random_subset(input_file_path, output_file_path, percentage=0.001):
    """
    从JSONL文件中随机抽取一定比例的行创建新的数据集
    
    参数:
        input_file_path: 输入的JSONL文件路径
        output_file_path: 输出的JSONL文件路径
        percentage: 要抽取的比例，默认是0.01（即1%）
    """
    # 读取所有行并计算总行数
    with open(input_file_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    
    total_lines = len(all_lines)
    sample_size = max(1, int(total_lines * percentage))  # 至少抽取1行
    
    print(f"文件总行数: {total_lines}")
    print(f"将抽取 {sample_size} 行 ({percentage*100:.1f}%)")
    
    # 随机选择行的索引
    sampled_indices = random.sample(range(total_lines), sample_size)
    
    # 根据选择的索引提取行
    sampled_lines = [all_lines[i] for i in sorted(sampled_indices)]
    
    # 写入新文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.writelines(sampled_lines)
    
    print(f"已成功创建抽样数据集: {output_file_path}")
    print(f"抽取的行索引: {sorted(sampled_indices)}")

if __name__ == "__main__":
    # 获取当前文件所在的目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建输入和输出文件的路径
    input_file = "/Users/tangbohao/Files/科研资料/EHR-decision/mimic-code/ehr_freetext/reconstruction_item_max.jsonl"
    output_file = os.path.join(os.path.dirname(input_file), "reconstruction_item_0.001.jsonl")
    
    # 执行抽样
    extract_random_subset(input_file, output_file, 0.001)