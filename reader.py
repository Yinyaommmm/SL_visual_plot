from typing import Literal
import pandas as pd

def read_excel(type:Literal["loss", "err"] ="loss"):
    # Read the data from Excel
    df = pd.read_excel('data.xlsx')

    # Drop the useless column and adjust columns
    df = df.drop(df.columns[0], axis=1)
    df = df.drop(df.columns[3], axis=1)  # After drop 3rd, the 4th become 3rd
    if type == "loss":
        df = df.drop(df.columns[3], axis=1)  # After drop 3rd, the 4th become 3rd
    elif type == "err":
        df = df.drop(df.columns[4],axis =1) 

    # Ensure the dataframe has the correct columns
    df = df.iloc[:, :4]

    # Rename columns for easier access
    df.columns = ['up_D', 'up_M', 'down_D', 'loss']
    return df

