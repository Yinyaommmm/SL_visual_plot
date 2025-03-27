import sys
import os
# 获取当前脚本所在目录的上一级目录（SL_visual_plot）,并添加为系统路径，方便引入util工具包
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from util.formatter import format2KorM, format2KorM_no100K,export_result
from util.reader import read_excel
from config import FIG_HEIGHT

df = read_excel()  # 读取数据，df.columns = ['up_D', 'up_M', 'down_D', 'loss']
# 按 'up_D' 进行分组
grouped = df.groupby('up_D')

custom_colors = [
    (0/255, 255/255, 255/255),    # 亮青色
    (51/255, 204/255, 255/255),   # 亮蓝色
    (102/255, 153/255, 255/255),  # 浅蓝色
    (100/255, 100/255, 255/255),  # 柔和蓝色
    # (153/255, 102/255, 255/255),  # 浅紫色
    # (204/255, 51/255, 255/255),   # 品红色
    # (255/255, 0/255, 255/255),    # 重品红色
    # (255/255, 0/255, 128/255),    # 深粉色
]

num_groups = len(grouped)  # 计算 'up_D' 组的数量

fig, axes = plt.subplots(1, num_groups, figsize=(5 * num_groups, FIG_HEIGHT), sharey=True)
fig.subplots_adjust(wspace=0.05)  # 调整子图间距

if num_groups == 1:
    axes = [axes]

for i, (ax, (up_D, group)) in enumerate(zip(axes, grouped)):
    color_cycle = cycle(custom_colors)
    for up_M, sub_group in group.groupby('up_M'):
        color = next(color_cycle)
        ax.plot(sub_group['down_D'], sub_group['loss'], label=f'Model Params={format2KorM(up_M)}', linestyle='--', marker='o', color=color)
    
    ax.set_title(f'Pretraining Data Size = {format2KorM(up_D)}', fontsize=18)
    
    min_x = group['down_D'].min()
    max_x = group['down_D'].max()
    margin = (max_x - min_x) * 0.05
    x_min = (min_x - margin) // 1000
    x_max = (max_x + margin) // 1000
    x_min, x_max = int(np.floor(x_min / 5) * 5), int(np.ceil(x_max / 5) * 5)
    x_min, x_max = x_min * 1000, x_max * 1000
    xticks = np.arange(x_min, x_max + 1, step=10000)
    
    ax.set_xlim(x_min, x_max)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{tick // 1000}' for tick in xticks])
    ax.legend(loc='upper right', fontsize=14)

fig.text(0.5, 0.02, 'Finetuning Data Size (K)', ha='center', fontsize=18)
axes[0].set_ylabel('$Cross\ Entropy\ Loss$', fontsize=18)


filename = os.path.splitext(os.path.basename(__file__))[0]
extension = 'pdf'
export_result(plt,f'./image/{filename}',extension)
