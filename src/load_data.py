import pandas as pd
import os

def load_all_csv(data_folder="data"):
    filenames = [
        "산림청_산불상황관제시스템 산불통계데이터_20241016.csv"
    ]

    dataframes = []
    for name in filenames:
        path = os.path.join(data_folder, name)
        try:
            df = pd.read_csv(path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='cp949')
        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)
