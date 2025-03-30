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
from config import FIG_HEIGHT,FIG_GAP


df = read_excel(type="error")  # 读取数据，df.columns = ['up_D', 'up_M', 'down_D', 'error']
# Group by 'up_M'
grouped = df.groupby('up_M')

custom_colors = [
    (0/255, 255/255, 255/255),    # light cyan
    (51/255, 204/255, 255/255),   # bright cyan
    (102/255, 153/255, 255/255),  # light blue
    (100/255, 100/255, 255/255),  # pastel blue
    (153/255, 102/255, 255/255),  # light purple
    (204/255, 51/255, 255/255),   # magenta
    (255/255, 0/255, 255/255),    # heavy magenta
    (255/255, 0/255, 128/255),    # deep pink
]

num_groups = len(grouped)

fig, axes = plt.subplots(1, num_groups, figsize=(5 * num_groups, FIG_HEIGHT), sharey=True)
fig.subplots_adjust(wspace=FIG_GAP)
if num_groups == 1:
    axes = [axes]

for i, (ax, (up_M, group)) in enumerate(zip(axes, grouped)):
    color_cycle = cycle(custom_colors)
    for down_D, sub_group in group.groupby('down_D'):
        color = next(color_cycle)
        ax.plot(sub_group['up_D'], sub_group['error'], label=f'Finetuning Data Size={format2KorM( down_D)}', linestyle='--', marker='o', color=color)
    
    ax.set_title(f'Model Params = {format2KorM(up_M)}', fontsize=18)
    
    min_x = group['up_D'].min()
    max_x = group['up_D'].max()
    margin = (max_x - min_x) * 0.05
    scale = 1000
    x_min = (min_x - margin) // scale
    x_max = (max_x + margin) // scale
    x_min, x_max = int(np.floor(x_min / 5) * 5), int(np.ceil(x_max / 5) * 5)
    x_min, x_max = x_min * scale, x_max * scale
    xticks = np.arange(x_min, x_max + 1, step=200000)
    print(xticks)
    ax.set_xlim(x_min, x_max)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{tick / 1000000:.1f}' for tick in xticks])

    ax.legend(loc='upper right', fontsize=12)


fig.text(0.5, 0.03, 'Pretraining Data Size (M)', ha='center', fontsize=18)
axes[0].set_ylabel('$Error$', fontsize=18)

filename = os.path.splitext(os.path.basename(__file__))[0]
extension = 'pdf'
export_result(plt,f'./image/{filename}',extension)