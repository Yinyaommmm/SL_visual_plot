import sys
import os
# 获取当前脚本所在目录的上一级目录（SL_visual_plot）,并添加为系统路径，方便引入util工具包
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from util.formatter import format2KorM, format2KorM_no100K,export_result
from util.reader import read_excel

df = read_excel(type="error") # df.columns is ['up_D', 'up_M', 'down_D', 'error']
# Group by 'up_M'
grouped = df.groupby('up_M')

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

# Number of unique up_M groups
num_groups = len(grouped)

# Create subplots: one row, `num_groups` columns, sharing the y-axis
fig, axes = plt.subplots(1, num_groups, figsize=(5 * num_groups, 6), sharey=True)  # sharey=True ensures they share the same y-axis
fig.subplots_adjust(wspace=0.05)  
# If there's only one group, axes will be a single object, not an array
if num_groups == 1:
    axes = [axes]

for i, (ax, (up_M, group)) in enumerate(zip(axes, grouped)):
    color_cycle = cycle(custom_colors)
    for up_D, sub_group in group.groupby('up_D'):
        color = next(color_cycle)
        ax.plot(sub_group['down_D'], sub_group['error'], label=f'up_D={up_D}', linestyle='--', marker='o', color=color)
    
    ax.set_title(f'$M = ${format2KorM(up_M)}', fontsize=18)
    # ax.grid(True) 开启网格
    # 计算 x 轴范围并添加 5% 边距
    min_x = group['down_D'].min()
    max_x = group['down_D'].max()
    margin = (max_x - min_x) * 0.05
    x_min = (min_x - margin) //1000
    x_max = (max_x + margin)//1000
    # 取整到 5 的倍数
    x_min,x_max = int(np.floor(x_min / 5) * 5), int(np.ceil(x_max / 5) * 5)
    print(x_min,x_max),
    x_min,x_max = x_min * 1000, x_max*1000
    # 生成 5 个等间距的整数刻度
    xticks = np.arange(x_min, x_max + 1, step=10000) 
    # 设置 x 轴范围
    ax.set_xlim(x_min, x_max)

    # 应用新的 x 轴刻度
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{tick // 1000}' for tick in xticks])


# Set the x-axis label for the entire figure
fig.text(0.5, 0.03, '$D_f$ (K)', ha='center', fontsize=18)

# Set y-axis label for the first subplot only
axes[0].set_ylabel('$Error$', fontsize=18)

# Determine the minimum and maximum values of 'up_D' across all groups
min_up_D = df['up_D'].min()
max_up_D = df['up_D'].max()

unique_up_D = np.sort(df['up_D'].unique())
# 创建一个 Normalize 对象，使颜色映射到 up_D 取值范围
norm = Normalize(vmin=unique_up_D.min(), vmax=unique_up_D.max()) 
cmap = colormaps['cool'] 

# Use a ScalarMappable for the colorbar with continuous normalization
cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation='vertical', fraction=0.03, pad=0.02)

# Set ticks only for the min and max up_D values
cbar.set_ticks(unique_up_D)  # Set ticks at the min and max values only
cbar.set_label('$D_p$', rotation=90, labelpad=10, fontsize=18)  # Set the label to 'up_D' as requested

# Optionally, set tick labels as min and max up_D
cbar.ax.set_yticklabels([format2KorM_no100K(val) for val in unique_up_D])  # Format the labels to two decimal places

# Save the figure to a file
export_result(plt,'./image/err_single_colorbar_rM_xDf',"pdf")
