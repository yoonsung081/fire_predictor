

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import json

# 모델 로드
model = joblib.load("C:/Users/000/OneDrive/Desktop/fire_predictor_project/models/damage_regression_model.joblib")

# 예측할 데이터 로드
rf_pred_df = pd.read_csv("C:/Users/000/OneDrive/Desktop/fire_predictor_project/data/rf_predict.csv")
with open("C:/Users/000/OneDrive/Desktop/fire_predictor_project/data/refined_predicted_fire_markers.json", 'r') as f:
    lgbm_pred_data = json.load(f)
lgbm_pred_df = pd.DataFrame(lgbm_pred_data)

pred_df = pd.concat([
    rf_pred_df.rename(columns={'LAT': 'lat', 'LON': 'lon'}),
    lgbm_pred_df
], ignore_index=True)
pred_df.rename(columns={'lat': 'LAT', 'lon': 'LON'}, inplace=True)

# 오늘 날짜 정보
today = datetime.now()
year = today.year
month = today.month
day = today.day
weekday = today.strftime("%A")

# 학습 데이터에서 가장 흔한 값 가져오기 (예시)
# 실제로는 학습 과정에서 저장해둔 값을 사용해야 합니다.
try:
    df_train = pd.read_csv("C:/Users/000/OneDrive/Desktop/fire_predictor_project/data/산림청_산불상황관제시스템 산불통계데이터_20241016.csv", encoding='utf-8')
except UnicodeDecodeError:
    df_train = pd.read_csv("C:/Users/000/OneDrive/Desktop/fire_predictor_project/data/산림청_산불상황관제시스템 산불통계데이터_20241016.csv", encoding='utf-8-sig')

most_common_values = {}
categorical_features = ['발생장소_관서', '발생장소_시도', '발생장소_시군구', '발생장소_읍면', '발생장소_동리', '발생원인_구분', '발생원인_세부원인']
for col in categorical_features:
    most_common_values[col] = df_train[col].mode()[0]

# 예측 수행
predictions = []
for _, row in pred_df.iterrows():
    feature_vector = {
        '발생일시_년': year,
        '발생일시_월': month,
        '발생일시_일': day,
        '발생일시_요일': weekday,
        'LAT': row['LAT'],
        'LON': row['LON'],
        **most_common_values
    }
    
    # DataFrame으로 변환 후 범주형 데이터 처리
    feature_df = pd.DataFrame([feature_vector])
    for col in categorical_features:
        feature_df[col] = feature_df[col].astype('category')
        # 학습 데이터에 없던 카테고리가 있을 경우를 대비
        known_categories = df_train[col].astype('category').cat.categories
        feature_df[col] = feature_df[col].cat.set_categories(known_categories)

    # '발생일시_요일' 처리
    feature_df['발생일시_요일'] = feature_df['발생일시_요일'].astype('category')
    known_categories_weekday = df_train['발생일시_요일'].astype('category').cat.categories
    feature_df['발생일시_요일'] = feature_df['발생일시_요일'].cat.set_categories(known_categories_weekday)

    # 예측
    predicted_damage = model.predict(feature_df)
    
    predictions.append({
        'lat': row['LAT'],
        'lon': row['LON'],
        'predicted_damage': predicted_damage[0]
    })

# 결과를 JSON 파일로 저장
with open("C:/Users/000/OneDrive/Desktop/fire_predictor_project/data/damage_predictions.json", 'w', encoding='utf-8') as f:
    json.dump(predictions, f, ensure_ascii=False, indent=4)

print(f"{len(predictions)}개의 위치에 대한 피해 면적 예측을 완료하고 'data/damage_predictions.json'에 저장했습니다.")

