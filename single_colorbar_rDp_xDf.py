import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from matplotlib.cm import ScalarMappable
from reader import read_excel
from formatter import format2KorM, format2KorM_no100K,export_result
from matplotlib.colors import ListedColormap, BoundaryNorm
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

fig, axes = plt.subplots(1, num_groups, figsize=(5 * num_groups, 6), sharey=True)
fig.subplots_adjust(wspace=0.05)  # 调整子图间距

if num_groups == 1:
    axes = [axes]

for i, (ax, (up_D, group)) in enumerate(zip(axes, grouped)):
    color_cycle = cycle(custom_colors)
    for up_M, sub_group in group.groupby('up_M'):
        color = next(color_cycle)
        ax.plot(sub_group['down_D'], sub_group['loss'], label=f'up_M={format2KorM(up_M)}', linestyle='--', marker='o', color=color)
    
    ax.set_title(f'$D_p = ${format2KorM(up_D)}', fontsize=18)
    
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

fig.text(0.5, 0.03, '$D_f$ (K)', ha='center', fontsize=18)
axes[0].set_ylabel('$Cross\ Entropy\ Loss$', fontsize=18)

# 创建colorbar
unique_up_M = np.sort(df['up_M'].unique())
# 创建一个仅包含 custom_colors 的 colormap
cmap = ListedColormap(custom_colors)

# 使用 unique_up_M 的实际数值作为 boundaries
boundaries = np.append(unique_up_M, unique_up_M[-1] + 1)  # 添加一个最大边界

# 创建一个 BoundaryNorm，使 colorbar 在 up_M 的范围内精确对应 custom_colors
norm = BoundaryNorm(boundaries, cmap.N, extend='neither')

# 创建 colorbar
cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation='vertical', fraction=0.03, pad=0.02)

# 使 colorbar 的 ticks 只出现在 unique_up_M 上
cbar.set_ticks(unique_up_M)
cbar.set_ticklabels([format2KorM_no100K(val) for val in unique_up_M])

cbar.set_label('$M$', rotation=90, labelpad=10, fontsize=18)

export_result(plt,'./image/single_colorbar_rDp_xDf',"pdf")
