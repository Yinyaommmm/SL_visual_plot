import sys
import os
from typing import Literal
# 获取当前脚本所在目录的上一级目录（SL_visual_plot）,并添加为系统路径，方便引入util工具包
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from  util.formatter import format2KorM_no100K

headMap = {
    2 : '2.5M',
    4 : '9.8M',
    6 : '21M',
    8 : '38M'
}
def read_excel( sheet_name: str = Literal["ImageNet100","TinyImageNet"]) -> pd.DataFrame:
    """
    读取 Excel 文件中的指定 Sheet，并返回 DataFrame。
    """
    df = pd.read_excel('./tobest.xlsx', sheet_name=sheet_name)
    return df

def draw(df: pd.DataFrame,sheet_name = Literal["ImageNet100","TinyImageNet"]):
    """
    绘制 5 组数据的柱形图，每组数据一张图。
    """
    # 获取唯一的指导组（Head2 -> Head6 等）
    groups = df[['up_M', 'teacher_M']].drop_duplicates().values.tolist()
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)
    title = [ f'using {headMap[x[0]]} model to distill {headMap[x[1]]} model'  for x in [[2,6],[4,6],[2,8],[4,8],[6,8]]]
    for i, (up_M, teacher_M) in enumerate(groups):
        ax = axes[i]
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        sub_df = df[(df['up_M'] == up_M) & (df['teacher_M'] == teacher_M)]
        
        # 横轴是 Pretraining Data 数量
        x = sub_df['up_D'].astype(int)
        x_label = [format2KorM_no100K(i) for i in x]
        x_index = range(len(x))
        
        # loss 对比
        width = 0.4  # 柱形宽度

        color_original = (135/255, 206/255, 235/255)  # 归一化 RGB 颜色
        color_method = (147/255, 112/255, 219/255)
        linewidth = 1.0
        ax.bar([idx - width/2 for idx in x_index], sub_df['ori_error'], width=width, label='Original', alpha=1, color=color_original,edgecolor='black',linewidth=linewidth)
        ax.bar([idx + width/2 for idx in x_index], sub_df['error'], width=width, label='Small to Large', alpha=1, color=color_method,edgecolor='black',linewidth=linewidth)
        
        ax.set_xticks(x_index)
        ax.set_xticklabels(x_label, rotation=45)
        ax.set_title(title[i],fontweight="bold")
        if i == 0:
            ax.set_ylabel('Error',fontsize=16)
        ax.legend()
    
    # fig.subplots_adjust(bottom=0.85)
    fig.text(0.5, 0.03, 'Pretraining Data', ha='center', fontsize=16)
    plt.tight_layout(pad=3.0)
    plt.suptitle("Error Comparison Across Different Heads and Pretraining Sizes", fontsize=15, fontweight='bold')
    fig.subplots_adjust(wspace=0.07)  # 调整子图间距
    plt.savefig(f'./image/{sheet_name}_distill.pdf')

# 示例用法
if __name__ == "__main__":
    sheet_name =  "ImageNet100"
    df = read_excel(sheet_name=sheet_name)
    draw(df,sheet_name)