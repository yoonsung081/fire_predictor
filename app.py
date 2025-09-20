import os
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import json
import torch
try:
    from tqdm import tqdm
except ImportError:
    print("'tqdm' 라이브러리가 설치되지 않았습니다. 'pip install tqdm'으로 설치해주세요.")
    # tqdm이 없을 경우, 일반 range를 사용하는 대체 함수
    def tqdm(iterable, *args, **kwargs):
        return iterable
import warnings

from src.geocoding import get_coordinates
from src.visualize import show_fire_map, show_long_term_prediction_map
from train_model import WildfireTransformer # 트랜스포머 모델 클래스 임포트
from sklearn.preprocessing import MinMaxScaler

LGBM_MODEL_PATH = "models/fire_predictor.joblib"
TRANSFORMER_MODEL_PATH = "models/transformer_predictor.pth"
SCALER_PATH = "models/transformer_scaler.joblib"

def run_historical_mode(df):
    """과거 산불 이력 조회 모드"""
    print("\n[날짜] 조회할 시작 날짜를 입력하세요 (예: 2023-03-01):")
    start_date_str = input("> ")
    print("🗓️ 조회할 종료 날짜를 입력하세요 (예: 2023-03-31):")
    end_date_str = input("> ")

    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    except ValueError:
        print("잘못된 날짜 형식입니다. YYYY-MM-DD 형식으로 입력해주세요.")
        return

    # pd.to_datetime이 인식할 수 있도록 컬럼명을 'year', 'month', 'day'로 변경합니다.
    datetime_cols = df[['발생일시_년', '발생일시_월', '발생일시_일']].copy()
    datetime_cols.columns = ['year', 'month', 'day']
    df['datetime'] = pd.to_datetime(datetime_cols, errors='coerce')
    df.dropna(subset=['datetime'], inplace=True)

    mask = (df['datetime'].dt.date >= start_date) & (df['datetime'].dt.date <= end_date)
    df_filtered = df.loc[mask]

    if not df_filtered.empty:
        print(f"\n✅ {start_date_str}부터 {end_date_str}까지 총 {len(df_filtered)}건의 산불이 조회되었습니다.")
        print("\n🗺️ 지도의 중심이 될 지역명을 입력하세요 (예: 대전):")
        region_name = input("> ")
        center_location = get_coordinates(region_name) or (36.5, 127.5)
        show_fire_map(center_location, df_filtered, f'{start_date_str} ~ {end_date_str} 산불 이력', '발생원인_세부원인', 'red')
    else:
        print(f"\nℹ️ 해당 기간({start_date_str} ~ {end_date_str})에는 산불 데이터가 없습니다.")

def run_short_term_prediction_mode(locations_df):
    """단기 예측 (LightGBM) 모드"""
    if not os.path.exists(LGBM_MODEL_PATH):
        print(f"🚨 단기 예측 모델({LGBM_MODEL_PATH})이 없습니다. train_lightgbm.py를 먼저 실행해주세요.")
        return

    model = joblib.load(LGBM_MODEL_PATH)
    print("\n🗓️ 예측할 날짜를 입력하세요 (예: 2024-10-26):")
    date_str = input("> ")
    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        print("잘못된 날짜 형식입니다. YYYY-MM-DD 형식으로 입력해주세요.")
        return

    print(f"📅 {date_str} 날짜의 예측 데이터 준비 중...")
    X_pred = locations_df[['LAT', 'LON']].copy()
    X_pred['월'] = target_date.month
    X_pred['요일'] = target_date.weekday()
    X_pred['피해면적_합계'] = 0  # 예측 시점에는 알 수 없으므로 0으로 설정
    X_pred = X_pred.fillna(0)

    print("🔥 단기 예측 수행 중...")
    print(f"전체 예측 대상 지역 수: {len(X_pred)}")
    
    # 배치 예측 및 프로그레스 바
    batch_size = 1000
    predictions = []
    pred_proba = []
    
    for i in tqdm(range(0, len(X_pred), batch_size), desc="LightGBM 예측"):
        batch = X_pred.iloc[i:i+batch_size]
        predictions.extend(model.predict(batch))
        pred_proba.extend(model.predict_proba(batch)[:, 1])

    predictions = np.array(predictions)
    pred_proba = np.array(pred_proba)

    result_df = locations_df.copy()
    result_df['fire_probability'] = pred_proba
    df_pred = result_df[predictions == 1]

    # 예측 경향 분석
    print("\n📊 예측 확률 분포 (Prediction Probability Distribution):")
    hist, bin_edges = np.histogram(pred_proba, bins=5)
    for i in range(len(hist)):
        print(f"  {bin_edges[i]:.2f} ~ {bin_edges[i+1]:.2f}: {hist[i]} 개 ({hist[i]/len(pred_proba)*100:.2f}%)")

    if not df_pred.empty:
        print(f"\n✅ {date_str}에 {len(df_pred)}곳의 산불 위험 지역이 예측되었습니다.")
        print("\n🗺️ 지도의 중심이 될 지역명을 입력하세요 (예: 강원):")
        region_name = input("> ")
        center_location = get_coordinates(region_name) or (37.5665, 126.9780)
        show_fire_map(center_location, df_pred, '단기 예측 결과', 'fire_probability', 'orange')
    else:
        print(f"\n✅ {date_str}에는 산불 위험이 예측된 지역이 없습니다.")

def run_long_term_prediction_mode(df, locations_df):
    """장기 예측 (Transformer) 모드"""
    if not all(os.path.exists(p) for p in [TRANSFORMER_MODEL_PATH, SCALER_PATH]):
        print(f"🚨 장기 예측 모델 또는 스케일러가 없습니다. train_transformer.py를 먼저 실행해주세요.")
        return

    # 모델과 스케일러 로드
    scaler = joblib.load(SCALER_PATH)
    features = ['LAT', 'LON', '월', '요일', '피해면적_합계', 'IS_FIRE']
    SEQ_LENGTH, PRED_LENGTH = 30, 7

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WildfireTransformer(n_features=len(features), n_heads=len(features), n_layers=2, dropout=0.1, pred_len=PRED_LENGTH).to(device)
    model.load_state_dict(torch.load(TRANSFORMER_MODEL_PATH, map_location=device))
    model.eval()

    print("\n⏳ 장기 예측을 위해 과거 30일치 데이터를 준비합니다 (가장 최신 데이터 사용).")
    # 실제로는 API 등을 통해 최신 30일 데이터를 가져와야 합니다. 여기서는 데모를 위해 학습 데이터의 마지막 부분을 사용합니다.
    day_map = {'월': 0, '화': 1, '수': 2, '목': 3, '금': 4, '토': 5, '일': 6}
    df['요일'] = df['발생일시_요일'].map(day_map)
    df_model_input = df[features].fillna(0).tail(SEQ_LENGTH * len(locations_df)) # 대략적인 지역별 데이터 추출
    
    # 지역별로 예측 수행
    print(f"🔥 미래 {PRED_LENGTH}일에 대한 장기 예측 수행 중...")
    all_preds = {}
    for _, location_info in locations_df.iterrows():
        # 해당 지역의 과거 데이터 시퀀스 생성 (데모용)
        # 실제로는 해당 지역의 정확한 과거 데이터를 사용해야 함
        loc_df = df_model_input.sample(SEQ_LENGTH, replace=True).copy()
        loc_df['LAT'] = location_info['LAT']
        loc_df['LON'] = location_info['LON']
        
        input_data_scaled = scaler.transform(loc_df[features])
        input_tensor = torch.FloatTensor(input_data_scaled).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            # 시그모이드 함수를 적용하여 확률로 변환
            probabilities = torch.sigmoid(output).squeeze().cpu().numpy()

        for i in range(PRED_LENGTH):
            day_key = f'Day {i+1}'
            if probabilities[i] > 0.5: # 임계값 0.5 이상일 때 위험으로 판단
                if day_key not in all_preds:
                    all_preds[day_key] = []
                loc_info['fire_probability'] = probabilities[i]
                all_preds[day_key].append(loc_info.copy())

    pred_dfs = {day: pd.DataFrame(preds) for day, preds in all_preds.items()}

    if not pred_dfs:
        print(f"\n✅ 향후 {PRED_LENGTH}일 동안 산불 위험이 예측된 지역이 없습니다.")
        return

    print(f"\n✅ 향후 {PRED_LENGTH}일 동안의 산불 위험 예측이 완료되었습니다.")
    print("\n🗺️ 지도의 중심이 될 지역명을 입력하세요 (예: 전국):")
    region_name = input("> ")
    center_location = get_coordinates(region_name) or (36.5, 127.5) # Default to center of Korea
    show_long_term_prediction_map(center_location, pred_dfs)

def show_model_performance():
    """모델 성능 지표 비교 대시보드"""
    metrics_path = "static/metrics.json"
    if not os.path.exists(metrics_path):
        print(f"\n🚨 성능 지표 파일({metrics_path})이 없습니다. 모델을 먼저 훈련시켜주세요.")
        return

    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    print("\n========================================")
    print("* 모델 성능 비교 대시보드 *")
    print("========================================")

    data_for_df = []
    
    # LightGBM (baseline) 데이터 추가
    if 'baseline' in metrics:
        lgbm_metrics = metrics['baseline']
        lgbm_metrics['Model'] = 'LightGBM (Baseline)'
        data_for_df.append(lgbm_metrics)

    # Transformer 데이터 추가
    if 'transformer' in metrics:
        transformer_metrics = metrics['transformer']
        transformer_metrics['Model'] = 'Transformer'
        data_for_df.append(transformer_metrics)

    if not data_for_df:
        print("ℹ️  표시할 성능 지표가 없습니다. 모델을 훈련하고 다시 시도해주세요.")
        return

    # pandas DataFrame을 사용하여 깔끔한 표로 출력
    df_metrics = pd.DataFrame(data_for_df).set_index('Model')
    
    # 소수점 4자리까지 포맷팅
    for col in ['accuracy', 'precision', 'recall', 'f1']:
        if col in df_metrics.columns:
            df_metrics[col] = df_metrics[col].apply(lambda x: f"{x:.4f}")
            
    print(df_metrics.to_string())

if __name__ == "__main__":
    data_path = "data/with_coordinates.csv"
    if not os.path.exists(data_path):
        print(f"🚨 데이터 파일({data_path})이 없습니다. 데이터를 먼저 준비해주세요.")
        exit(1)

    print("- 산불 이력 데이터를 로딩합니다...")
    df = pd.read_csv(data_path)
    locations_df = df[['LAT', 'LON', 'full_address']].drop_duplicates().reset_index(drop=True)
    df['IS_FIRE'] = (df['피해면적_합계'] > 0).astype(int)

    while True:
        print("\n========================================")
        print("* 산불 예측 및 조회 시스템 *")
        print("========================================")
        print("1: 과거 산불 이력 조회")
        print("2: 단기 산불 위험 예측 (LightGBM)")
        print("3: 장기 산불 위험 예측 (Transformer)")
        print("4: 모델 성능 비교")
        print("q: 종료")
        print("----------------------------------------")
        mode = input("원하는 작업의 번호를 입력하세요 > ")

        if mode == '1':
            run_historical_mode(df.copy())
        elif mode == '2':
            run_short_term_prediction_mode(locations_df.copy())
        elif mode == '3':
            run_long_term_prediction_mode(df.copy(), locations_df.copy())
        elif mode == '4':
            show_model_performance()
        elif mode.lower() == 'q':
            print("프로그램을 종료합니다.")
            break
        else:
            print("🚨 잘못된 입력입니다. 1, 2, 3, 4, q 중에서 선택해주세요.")
