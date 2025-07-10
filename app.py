import os
import pandas as pd
from src.load_data import load_all_csv
from src.geocoding import add_lat_lon_from_address, get_coordinates
from src.model import (
    add_fire_label,
    train_and_predict_weather,
    train_and_predict_baseline
)
from src.visualize import show_comparison_map

if __name__ == "__main__":
    cache_path = "data/with_coordinates.csv"

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
    else:
        df = load_all_csv()
        df = add_lat_lon_from_address(df)
        df.to_csv(cache_path, index=False)

    df = add_fire_label(df)

    print("검색할 지역명을 입력하세요:")
    region = input("> ")

    center_location = get_coordinates(region)
    if not center_location:
        print(f"'{region}' 위치를 찾을 수 없습니다.")
        exit(1)

    df_true, df_pred_weather, df_pred_baseline = (
        train_and_predict_weather(df),
        train_and_predict_baseline(df)
    )

    show_comparison_map(center_location, df_true, df_pred_weather, df_pred_baseline)
