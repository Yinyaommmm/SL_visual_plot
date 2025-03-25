import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from reader import read_excel
from formatter import format2KorM, format2KorM_no100K,export_result

df = read_excel()  # df.columns is ['up_D', 'up_M', 'down_D', 'loss']
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

fig, axes = plt.subplots(1, num_groups, figsize=(5 * num_groups, 6), sharey=True)
fig.subplots_adjust(wspace=0.05)
if num_groups == 1:
    axes = [axes]

for i, (ax, (up_M, group)) in enumerate(zip(axes, grouped)):
    color_cycle = cycle(custom_colors)
    for down_D, sub_group in group.groupby('down_D'):
        color = next(color_cycle)
        ax.plot(sub_group['up_D'], sub_group['loss'], label=f'down_D={down_D}', linestyle='--', marker='o', color=color)
    
    ax.set_title(f'$M = ${format2KorM(up_M)}', fontsize=18)
    
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

fig.text(0.5, 0.03, '$D_p$ (M)', ha='center', fontsize=18)
axes[0].set_ylabel('$Cross\ Entropy\ Loss$', fontsize=18)

unique_down_D = np.sort(df['down_D'].unique())
norm = Normalize(vmin=unique_down_D.min(), vmax=unique_down_D.max())
cmap = colormaps['cool']

cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation='vertical', fraction=0.03, pad=0.02)
cbar.set_ticks(unique_down_D)
cbar.set_label('$D_f$', rotation=90, labelpad=10, fontsize=18)
cbar.ax.set_yticklabels([format2KorM_no100K(val) for val in unique_down_D])

export_result(plt,'./image/single_colorbar_rM_xDp',"pdf")