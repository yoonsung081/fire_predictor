import pandas as pd
import os

def load_all_csv(data_folder="data"):
    filenames = [
        "산림청_산불상황관제시스템 산불통계데이터_20241016.csv"
    ]

    dataframes = []
    # data_folder 내의 모든 파일 목록을 가져옵니다.
    all_files_in_data_folder = os.listdir(data_folder)

    for name in filenames:
        found_file = False
        for actual_file_name in all_files_in_data_folder:
            # 파일 이름이 일치하는지 확인 (대소문자 구분 없이)
            if name.lower() == actual_file_name.lower():
                path = os.path.join(data_folder, actual_file_name)
                try:
                    df = pd.read_csv(path, encoding='utf-8-sig')
                except UnicodeDecodeError:
                    df = pd.read_csv(path, encoding='cp949')
                dataframes.append(df)
                found_file = True
                break
        if not found_file:
            print(f"Warning: Expected file '{name}' not found in '{data_folder}'.")
        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)
