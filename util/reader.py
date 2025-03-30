from typing import Literal
import pandas as pd
import os 
def read_excel(type:Literal["loss", "error"] ="loss",dataset:Literal["CIFAR10","CIFAR100","TinyImageNet","ImageNet100"] = "ImageNet100"):
    # Read the data from Excel
    Base_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_excel(os.path.join(Base_DIR,'data.xlsx'),sheet_name=dataset)

    # Drop the useless column and adjust columns
    df = df.drop(df.columns[0], axis=1)
    df = df.drop(df.columns[3], axis=1)  # After drop 3rd, the 4th become 3rd
    if type == "loss":
        df = df.drop(df.columns[3], axis=1)  # After drop 3rd, the 4th become 3rd
    elif type == "error":
        df = df.drop(df.columns[4],axis =1) 

    # Ensure the dataframe has the correct columns
    df = df.iloc[:, :4]

    # Rename columns for easier access
    df.columns = ['up_D', 'up_M', 'down_D', type]
    return df

