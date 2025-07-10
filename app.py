import os
import pandas as pd
from src.load_data import load_all_csv
from src.visualize import show_map_with_prediction
from src.geocoding import add_lat_lon_from_address
from src.model import train_and_predict, add_fire_label

if __name__ == "__main__":
    cache_path = "data/with_coordinates.csv"

    if os.path.exists(cache_path):
        print("✅ 좌표 캐시 불러오는 중...")
        df = pd.read_csv(cache_path)
    else:
        print("📦 좌표 변환 중...")
        df = load_all_csv()
        df = add_lat_lon_from_address(df)
        df.to_csv(cache_path, index=False)
        print("✅ 좌표 캐시 저장 완료")

    df = add_fire_label(df)
    prediction_df = train_and_predict(df)
    show_map_with_prediction(df, prediction_df)
