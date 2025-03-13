import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
with open('/Users/tangbohao/Desktop/mimic/SFT/dataset.json', 'r') as f:
    data = json.load(f)

# 统计每个数字出现的次数
counter = Counter(data)
x = sorted(counter.keys())
y = [counter[i] for i in x]

# 找出需要标注的点
max_times_index = y.index(max(y))
max_times_point = (x[max_times_index], y[max_times_index])
min_item_point = (min(x), counter[min(x)])
max_item_point = (max(x), counter[max(x)])

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_context("notebook")

# 创建曲线图
plt.figure(figsize=(12, 6))
plt.plot(x, y, '-', linewidth=2, color='blue')

# 添加标注点
plt.scatter([max_times_point[0]], [max_times_point[1]], color='red', zorder=5)
plt.scatter([min_item_point[0]], [min_item_point[1]], color='green', zorder=5)
plt.scatter([max_item_point[0]], [max_item_point[1]], color='orange', zorder=5)

# 添加标注文本
plt.annotate(f'({max_times_point[0]}, {max_times_point[1]})', 
            xy=max_times_point, xytext=(10, 10), 
            textcoords='offset points', fontsize=14, color='red')
plt.annotate(f'({min_item_point[0]}, {min_item_point[1]})', 
            xy=min_item_point, xytext=(10, 10), 
            textcoords='offset points', fontsize=14, color='red')
plt.annotate(f'({max_item_point[0]}, {max_item_point[1]})', 
            xy=max_item_point, xytext=(10, -20), 
            textcoords='offset points', fontsize=14, color='red')

# 在y=20处添加水平参考线
plt.axhline(y=20, color='red', linestyle='--', alpha=0.5)

# 找出交点
intersect_x = []
intersect_y = []
for i in range(len(x)-1):
    if (y[i] - 20) * (y[i+1] - 20) < 0:  # 说明交点在这两点之间
        # 使用线性插值计算交点x坐标
        x_intersect = x[i] + (x[i+1] - x[i]) * (20 - y[i]) / (y[i+1] - y[i])
        intersect_x.append(x_intersect)
        intersect_y.append(20)

# 只保留最左边和最右边的交点
if len(intersect_x) >= 2:
    left_point = (min(intersect_x), 20)
    right_point = (max(intersect_x), 20)
    
    plt.scatter([left_point[0], right_point[0]], [20, 20], color='purple', zorder=5)
    plt.annotate(f'({int(left_point[0])}, 20)', 
                xy=left_point, xytext=(10, 10), 
                textcoords='offset points', fontsize=16, color='red')
    plt.annotate(f'({int(right_point[0])}, 20)', 
                xy=right_point, xytext=(10, 10), 
                textcoords='offset points', fontsize=16, color='red')
# 设置标题和标签
plt.title('item_num - times', fontsize=14)
plt.xlabel('item_num', fontsize=12)
plt.ylabel('times', fontsize=12)

# 优化布局
plt.tight_layout()

# 保存图片
plt.savefig('distribution_line.png', dpi=300, bbox_inches='tight')
plt.show()