import pandas as pd
from src.model import train_and_predict, add_fire_label
from src.visualize import find_nearest_peak

# ✅ 이미 위경도 포함된 파일 로딩
print("📦 with_coordinates.csv 불러오는 중...")
df = pd.read_csv("data/with_coordinates.csv")

# ✅ IS_FIRE 라벨 생성
df = add_fire_label(df)

# ✅ 예측
print("🧠 산불 예측 실행 중...")
pred_df = train_and_predict(df)

# ✅ 과거 산불 위치 보정
print("📍 과거 산불 위치 보정 중...")
df_actual_fixed = df.copy()
df_actual_fixed[['LAT', 'LON', 'MOUNTAIN']] = df_actual_fixed.apply(
    lambda row: pd.Series(find_nearest_peak(row['LAT'], row['LON'])),
    axis=1
)
df_actual_fixed.to_csv("data/fixed_actual.csv", index=False)
print("✅ fixed_actual.csv 저장 완료")

# ✅ 예측 산불 위치 보정
print("🔥 예측 산불 위치 보정 중...")
pred_df_fixed = pred_df.copy()
pred_df_fixed[['LAT', 'LON', 'MOUNTAIN']] = pred_df_fixed.apply(
    lambda row: pd.Series(find_nearest_peak(row['LAT'], row['LON'])),
    axis=1
)
pred_df_fixed.to_csv("data/fixed_predict.csv", index=False)
print("✅ fixed_predict.csv 저장 완료")
