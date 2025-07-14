import geopandas as gpd
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os

def calculate_and_update_metrics(true_fires_path, predicted_fires_path, metrics_file_path, model_name="rule_based_prediction"):
    print(f"✨ {model_name} 모델 지표 계산 및 업데이트 시작...")

    # 실제 산불 데이터 로드
    true_fires_gdf = gpd.read_file(true_fires_path)
    # '발생여부' 컬럼이 1이면 산불 발생, 0이면 미발생으로 가정 (데이터셋 구조에 따라 조정 필요)
    # 여기서는 '발생여부' 컬럼이 없으므로, 모든 지점이 산불 발생 지점이라고 가정하고 'is_fire'를 1로 설정
    # 실제 데이터셋에 '발생여부' 또는 유사한 컬럼이 있다면 해당 컬럼을 사용해야 합니다.
    true_fires_gdf['is_fire'] = 1 

    # 예측 결과 로드
    predicted_fires_gdf = gpd.read_file(predicted_fires_path)

    # 두 데이터프레임을 공간적으로 조인하거나, 공통 ID를 기반으로 병합
    # 여기서는 간단하게 인덱스를 기반으로 병합한다고 가정합니다.
    # 실제로는 지리적 위치나 고유 ID를 기반으로 정확히 매칭해야 합니다.
    merged_df = pd.merge(true_fires_gdf, predicted_fires_gdf[['FIRE_RISK', 'geometry']], how='left', on='geometry')

    # 결측치 처리 (예측 결과가 없는 경우 0으로 간주)
    merged_df['FIRE_RISK'] = merged_df['FIRE_RISK'].fillna(0)

    # 실제 값과 예측 값 추출
    y_true = merged_df['is_fire']
    y_pred = merged_df['FIRE_RISK']

    # 지표 계산
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    print(f"계산된 지표: {metrics}")

    # 기존 metrics.json 파일 로드
    if os.path.exists(metrics_file_path):
        with open(metrics_file_path, 'r', encoding='utf-8') as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    # 새로운 모델 지표 추가 또는 업데이트
    all_metrics[model_name] = metrics

    # 업데이트된 metrics.json 저장
    with open(metrics_file_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    print(f"지표가 {metrics_file_path}에 성공적으로 업데이트되었습니다.")

if __name__ == "__main__":
    true_fires_path = "data/true_fires.geojson"
    predicted_fires_path = "data/true_fires_with_weather_risk.geojson"
    metrics_file_path = "static/metrics.json"
    
    calculate_and_update_metrics(true_fires_path, predicted_fires_path, metrics_file_path, model_name="satellite_rule_based")
