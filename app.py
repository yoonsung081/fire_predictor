import os
import pandas as pd
from src.visualize import show_map_with_prediction
from src.geocoding import add_lat_lon_from_address, filter_to_korea
from src.model import train_and_predict, add_fire_label

if __name__ == "__main__":
    raw_path = "data/산림청_산불상황관제시스템 산불통계데이터_20241016.csv"
    cache_path = "data/with_coordinates.csv"

    if os.path.exists(cache_path):
        print("✅ 캐시된 좌표 파일 불러오는 중...")
        df = pd.read_csv(cache_path, encoding='utf-8-sig')
    else:
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"원본 CSV 없음: {raw_path}")
        print("✅ 산림청 산불 데이터 불러오는 중...")
        df = pd.read_csv(raw_path, encoding='utf-8-sig')
        df = add_lat_lon_from_address(df)
        df = filter_to_korea(df)
        df.to_csv(cache_path, index=False, encoding='utf-8-sig')
        print("✅ 변환된 좌표 캐시에 저장 완료!")

    df = add_fire_label(df)
    prediction_df = train_and_predict(df)
    show_map_with_prediction(df, prediction_df)