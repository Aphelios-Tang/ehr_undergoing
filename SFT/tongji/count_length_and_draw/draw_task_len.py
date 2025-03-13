import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_distribution(key_name):
    # 读取数据
    with open('task_length.json', 'r') as f:
        data = json.load(f)
    
    if key_name not in data:
        print(f"Key {key_name} 不存在")
        return
        
    values = np.array(data[key_name])
    values = values / 4 # token长度
    
    # 设置样式
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    
    
    # KDE分布图
    sns.kdeplot(
        data=values,
        fill=True,
        color='skyblue',
        alpha=0.5,
        label='KDE Curve'
    )
    
    
    # 添加统计信息
    mean_val = np.mean(values)
    median_val = np.median(values)
    std_val = np.std(values)
    
    plt.axvline(mean_val, color='red', linestyle='--', label=f'mean: {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='--', label=f'median num: {median_val:.2f}')

    # 添加3σ范围
    plt.axvline(mean_val + std_val, color='purple', linestyle=':', 
                label=f'μ+σ: {mean_val + std_val:.2f}')
    plt.axvline(mean_val - std_val, color='purple', linestyle=':',
                label=f'μ-σ: {mean_val - std_val:.2f}')
    
    # 填充3σ区域
    plt.axvspan(mean_val - std_val, mean_val + std_val, 
                alpha=0.2, color='yellow', label=f'σ range (σ={std_val:.2f})')
    
    # 设置图表
    plt.title(f'{key_name} input + instruction len distribution')
    plt.xlabel(f'len of {key_name} (token)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    # 保存
    plt.savefig("fig_task_len/" + key_name + "_len_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

with open('task_length.json', 'r') as f:
    data = json.load(f)
    for key in data.keys():
        plot_distribution(key)
    