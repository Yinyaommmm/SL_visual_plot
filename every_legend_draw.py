import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle

# Read the data from Excel
df = pd.read_excel('data.xlsx')

# Drop the useless column and adjust columns
df = df.drop(df.columns[0], axis=1)
df = df.drop(df.columns[3], axis=1)  # After drop 3rd, the 4th become 3rd
df = df.drop(df.columns[3], axis=1)  # After drop 3rd, the 4th become 3rd

# Ensure the dataframe has the correct columns
df = df.iloc[:, :4]

# Rename columns for easier access
df.columns = ['up_D', 'up_M', 'down_D', 'loss']

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

# If there's only one group, axes will be a single object, not an array
if num_groups == 1:
    axes = [axes]

# Iterate over each group (i.e., each up_M value)
for i, (ax, (up_M, group)) in enumerate(zip(axes, grouped)):
    color_cycle = cycle(custom_colors)
    for up_D, sub_group in group.groupby('up_D'):
        # Get the next color in the cycle
        color = next(color_cycle)
        ax.plot(sub_group['down_D'], sub_group['loss'], label=f'up_D={up_D}', linestyle='--', marker='o', color=color)
    
    ax.set_xlabel('$D_f$ (K)')
    # Set y-axis label only for the first subplot
    if i == 0:
        ax.set_ylabel('Loss')
    else:
        ax.set_ylabel('')  # Remove y-label for other subplots
    
    ax.set_title(f'Loss vs down_D for up_M={up_M}')
    ax.grid(True)
    ax.legend()

# Adjust layout to avoid overlap
plt.tight_layout()

# Save the figure to a file
plt.savefig('./image/every_legend.png')
