import pandas as pd
from src.model import train_and_predict, add_fire_label
from src.visualize import find_nearest_peak
from src.load_data import load_all_csv
from src.geocoding import add_lat_lon_from_address
import os

print("📦 데이터 불러오는 중...")
cache_path = "data/with_coordinates.csv"

if os.path.exists(cache_path):
    df = pd.read_csv(cache_path)
else:
    df = load_all_csv()
    df = add_lat_lon_from_address(df)
    df.to_csv(cache_path, index=False)
    print("✅ 좌표 캐시 저장 완료")

df = add_fire_label(df)
pred_df = train_and_predict(df)

print("📍 과거 산불 위치 보정 중...")
df_actual_fixed = df.copy()
df_actual_fixed[['LAT', 'LON']] = df_actual_fixed[['LAT', 'LON']]  # 이미 LAT, LON 있으므로 단순 복사
df_actual_fixed.to_csv("data/fixed_actual.csv", index=False)
print("✅ fixed_actual.csv 저장 완료")

print("🔥 예측 산불 위치 보정 중...")
pred_df_fixed = pred_df.copy()
pred_df_fixed[['LAT', 'LON']] = pred_df_fixed[['LAT', 'LON']]
pred_df_fixed.to_csv("data/fixed_predict.csv", index=False)
print("✅ fixed_predict.csv 저장 완료")
