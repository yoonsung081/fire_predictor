
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime, timedelta

# --- Configuration ---
ACTUAL_DATA_PATH = "data/fixed_actual.csv"
MODEL_PATH = "models/fire_prediction_model.joblib"
OUTPUT_JSON_PATH = "data/daily_top_future_predictions.json"
START_DATE = "2026-04-02"
END_DATE = "2027-04-02"
TOP_PERCENTAGE = 0.15  # 전체 기간 중 상위 15%의 가장 위험한 날만 선정

def generate_future_data(start_date_str, end_date_str, actual_data_path):
    """Generates a DataFrame for future predictions."""
    if not os.path.exists(actual_data_path):
        raise FileNotFoundError(f"Actual data file not found: {actual_data_path}")
    actual_df = pd.read_csv(actual_data_path, encoding='utf-8-sig')
    
    locations = actual_df[['발생장소_시도', '발생장소_시군구', 'LAT', 'LON']].drop_duplicates(subset=['발생장소_시도', '발생장소_시군구'])
    
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    
    future_data = []
    for date in date_range:
        for hour in [10, 14, 18]:
            for _, loc_row in locations.iterrows():
                future_data.append({
                    'latitude': loc_row['LAT'],
                    'longitude': loc_row['LON'],
                    'month': date.month,
                    'day': date.day,
                    'dayofweek': date.weekday(),
                    'hour': hour,
                    'date_str': date.strftime('%Y%m%d'),
                    'location_sido': loc_row['발생장소_시도'],
                    'location_sigungu': loc_row['발생장소_시군구']
                })
    
    return pd.DataFrame(future_data)

if __name__ == "__main__":
    print(f"Generating future data for period: {START_DATE} to {END_DATE}")
    future_df = generate_future_data(START_DATE, END_DATE, ACTUAL_DATA_PATH)
    
    features_for_prediction = ['latitude', 'longitude', 'month', 'day', 'dayofweek', 'hour']
    X_future = future_df[features_for_prediction]
    
    print("Loading trained model...")
    model = joblib.load(MODEL_PATH)
    
    print("Calculating probabilities for all dates...")
    future_probabilities = model.predict_proba(X_future)[:, 1]
    future_df['FIRE_PROBABILITY'] = future_probabilities
    
    # 1. 각 날짜별로 가장 높은 확률값을 가진 행 하나씩만 추출
    daily_best = future_df.loc[future_df.groupby('date_str')['FIRE_PROBABILITY'].idxmax()].copy()
    
    # 2. 전체 날짜 중 상위 15%에 해당하는 확률 기준(Threshold) 계산
    num_days_to_select = int(len(daily_best) * TOP_PERCENTAGE)
    threshold_val = daily_best['FIRE_PROBABILITY'].sort_values(ascending=False).iloc[num_days_to_select]
    
    print(f"Dynamic Threshold for top {TOP_PERCENTAGE*100}%: {threshold_val:.4f}")
    
    # 3. 기준 이상의 날짜만 최종 선정
    high_risk_days = daily_best[daily_best['FIRE_PROBABILITY'] >= threshold_val].copy()

    # --- Save to JSON ---
    output_predictions = []
    for _, row in high_risk_days.iterrows():
        output_predictions.append({
            "date": row['date_str'],
            "lat": row['latitude'],
            "lon": row['longitude'],
            "probability": float(row['FIRE_PROBABILITY']),
            "location_sido": row['location_sido'],
            "location_sigungu": row['location_sigungu']
        })
        
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_predictions, f, ensure_ascii=False, indent=4)
        
    print(f"✅ 상위 {TOP_PERCENTAGE*100}%에 해당하는 {len(output_predictions)}일분의 고위험 예측 데이터가 저장되었습니다.")
    print(f"(전체 {len(daily_best)}일 중 {len(output_predictions)}일만 엄선됨)")
