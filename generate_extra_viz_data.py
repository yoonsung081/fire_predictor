
import pandas as pd
import numpy as np
import joblib
import json
import os

def generate_extra_data(data_path, model_path, output_path):
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    # 1 & 3: 원인 및 시간대 데이터 추출
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    
    # 시간 정보 처리 (HH:MM 형식일 경우 대비)
    if df['발생일시_시간'].dtype == object:
        df['hour'] = pd.to_datetime(df['발생일시_시간'], format='%H:%M', errors='coerce').dt.hour
    else:
        df['hour'] = df['발생일시_시간']
    
    # 1. 원인별 분포 (상위 7개 + 기타)
    cause_counts = df['발생원인_구분'].value_counts()
    top_causes = cause_counts.head(7).to_dict()
    if len(cause_counts) > 7:
        top_causes['기타'] = cause_counts.iloc[7:].sum()
    
    # 3. 시간대별 분포 (0-23시)
    hourly_counts = df['hour'].value_counts().sort_index().to_dict()
    # 모든 시간대가 채워지도록 보정
    full_hourly = {str(h): int(hourly_counts.get(h, 0)) for h in range(24)}

    # 4. 변수 중요도 (모델에서 추출)
    feature_importance = {}
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        # 학습시 사용된 피처 리스트 (train_and_save_model.py 기준)
        features = ['latitude', 'longitude', 'month', 'day', 'dayofweek', 'hour']
        importances = model.feature_importances_
        feature_importance = {feat: float(imp) for feat, imp in zip(features, importances)}
        # 중요도 순으로 정렬
        feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))

    # 결과 저장
    result = {
        "causes": top_causes,
        "hourly": full_hourly,
        "importance": feature_importance
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    print(f"Extra visualization data saved to {output_path}")

if __name__ == "__main__":
    # 데이터 경로 확인 (fixed_actual.csv 또는 원본 데이터)
    data_path = "data/fixed_actual.csv"
    model_path = "models/fire_prediction_model.joblib"
    output_path = "static/extra_viz_data.json"
    generate_extra_data(data_path, model_path, output_path)
