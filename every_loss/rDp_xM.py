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


df = read_excel() # df.columns is ['up_D', 'up_M', 'down_D', 'loss']
# Group by 'up_D'
grouped = df.groupby('up_D')

custom_colors = [
    (0/255, 255/255, 255/255),    # rgb(0,255,255) - light cyan
    (51/255, 204/255, 255/255),   # rgb(51,204,255) - bright cyan
    (102/255, 153/255, 255/255),  # rgb(102,153,255) - light blue
    (100/255, 100/255, 255/255),  # rgb(100,100,255) - pastel blue
    (153/255, 102/255, 255/255),  # rgb(153,102,255) - light purple
    (204/255, 51/255, 255/255),   # rgb(204,51,255) - magenta
    (255/255, 0/255, 255/255),    # rgb(255,0,255) - heavy magenta
    (255/255, 0/255, 128/255),    # rgb(255,0,128) - deep pink
]

# Number of unique up_D groups
num_groups = len(grouped)

# Create subplots: one row, `num_groups` columns, sharing the y-axis
fig, axes = plt.subplots(1, num_groups, figsize=(5 * num_groups, FIG_HEIGHT), sharey=True)  # sharey=True ensures they share the same y-axis
fig.subplots_adjust(wspace=0.05)  
# If there's only one group, axes will be a single object, not an array
if num_groups == 1:
    axes = [axes]

for i, (ax, (up_D, group)) in enumerate(zip(axes, grouped)):
    color_cycle = cycle(custom_colors)
    for down_D, sub_group in group.groupby('down_D'):
        color = next(color_cycle)
        ax.plot(sub_group['up_M'], sub_group['loss'], label=f'Finetuning Data Size={format2KorM(down_D)}', linestyle='--', marker='o', color=color)
    ax.set_title(f'Pretraining Data Size = {format2KorM(up_D)}', fontsize=18)
    
    # Compute x-axis range with 5% margin
    min_x = group['up_M'].min()
    max_x = group['up_M'].max()
    margin = (max_x - min_x) * 0.05
    scale = 100000
    x_min = (min_x - margin) // scale
    x_max = (max_x + margin) // scale
    
    # Round to multiples of 5
    x_min, x_max = int(np.floor(x_min / 5) * 5), int(np.ceil(x_max / 5) * 5)
    x_min, x_max = x_min * scale, x_max * scale
    
    # Generate 5 evenly spaced tick marks
    xticks = np.arange(x_min, x_max + 1, step=10000000)
    print(xticks)
    # Set x-axis range
    ax.set_xlim(x_min, x_max)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{tick // 1000000}' for tick in xticks])

    ax.legend(loc='upper right', fontsize=12)

# Set the x-axis label for the entire figure
fig.text(0.5, 0.03, 'Model Params (M)', ha='center', fontsize=18)

# Set y-axis label for the first subplot only
axes[0].set_ylabel('$Cross\ Entropy\ Loss$', fontsize=18)


filename = os.path.splitext(os.path.basename(__file__))[0]
extension = 'pdf'
export_result(plt,f'./image/{filename}',extension)