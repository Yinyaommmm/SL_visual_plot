import sys
import os
from typing import Literal
# 获取当前脚本所在目录的上一级目录（SL_visual_plot）,并添加为系统路径，方便引入util工具包
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import pandas as pd
import matplotlib.pyplot as plt
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
    
    fig, axes = plt.subplots(1, 5, figsize=(8, 2.2), sharey=True)
    title = [ f'{headMap[x[0]]} -> {headMap[x[1]]}'  for x in [[2,6],[4,6],[2,8],[4,8],[6,8]]]
    for i, (up_M, teacher_M) in enumerate(groups):
        ax = axes[i]
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        sub_df = df[(df['up_M'] == up_M) & (df['teacher_M'] == teacher_M)]
        # 横轴是 Pretraining Data 数量
        x = sub_df['up_D'].astype(int)
        x_label = [format2KorM_no100K(i) for i in x]
        
        color_original = (135/255, 206/255, 235/255)  # 归一化 RGB 颜色
        color_method = (147/255, 112/255, 219/255)
        linewidth = 1.0

        axes[i].plot(sub_df['up_D'], sub_df['delta_error'], marker='o', linestyle='-',color=color_method,linewidth=linewidth,markersize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(x_label, rotation=45,fontsize=5)
        ax.set_title(title[i],fontsize=7)
        if i == 0:
            ax.set_ylabel('Delta Error',fontsize=9)
    
    # fig.subplots_adjust(bottom=0.85)
    fig.text(0.5, 0.03, 'Pretraining Data', ha='center', fontsize=8)
    plt.tight_layout(pad=1.4)
    plt.suptitle("The trend of (E1 - E2) across different pretraining data sizes ", fontsize=9)
    fig.subplots_adjust(wspace=0.07)  # 调整子图间距
    plt.savefig(f'./image/{sheet_name}_delta.pdf')

# 示例用法
if __name__ == "__main__":
    sheet_name =  "ImageNet100"
    df = read_excel(sheet_name=sheet_name)
    draw(df,sheet_name)