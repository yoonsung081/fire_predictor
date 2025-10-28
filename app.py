import os
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import json
import torch

# tqdm is a nice-to-have for progress bars
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

import config
from src.geocoding import get_coordinates
from src.visualize import export_data_to_json, open_dashboard
from src.prediction import predict_lgbm

# TODO: The WildfireTransformer class definition should be moved to a separate file in src/
# For now, we import it from train_model, but this is not ideal.
from train_model import WildfireTransformer # 트랜스포머 모델 클래스 임포트
from sklearn.preprocessing import MinMaxScaler

def run_historical_mode(df):
    """Exports historical fire data based on user-selected dates."""
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

    datetime_cols = df[['발생일시_년', '발생일시_월', '발생일시_일']].copy()
    datetime_cols.columns = ['year', 'month', 'day']
    df['datetime'] = pd.to_datetime(datetime_cols, errors='coerce')
    df.dropna(subset=['datetime'], inplace=True)

    mask = (df['datetime'].dt.date >= start_date) & (df['datetime'].dt.date <= end_date)
    df_filtered = df.loc[mask].copy()
    df_filtered['date'] = df_filtered['datetime'].dt.strftime('%Y-%m-%d')

    if not df_filtered.empty:
        print(f"\n✅ {start_date_str}부터 {end_date_str}까지 총 {len(df_filtered)}건의 산불 데이터를 JSON으로 저장합니다.")
        export_data_to_json(df_filtered, 'true_fires.json', extra_cols=['full_address', 'date'])
        open_dashboard()
    else:
        print(f"\nℹ️ 해당 기간({start_date_str} ~ {end_date_str})에는 산불 데이터가 없습니다.")

def run_short_term_prediction_mode():
    """Runs short-term prediction and exports the result to JSON."""
    print("\n🗓️ 단기 예측을 수행합니다.")
    
    data_path = config.ACTUAL_DATA_PATH 

    if not os.path.exists(data_path):
        print(f"🚨 데이터 파일({data_path})이 없습니다.")
        return

    df_pred = predict_lgbm(data_path, 'lgbm_predictions.json')

    if not df_pred.empty:
        print(f"\n✅ {len(df_pred)}곳의 산불 위험 지역이 예측되었습니다. 대시보드를 엽니다.")
        open_dashboard()
    else:
        print(f"\n✅ 산불 위험이 예측된 지역이 없습니다.")

def run_long_term_prediction_mode(df, locations_df):
    """Runs long-term prediction and exports the result to JSON."""
    if not all(os.path.exists(p) for p in [config.TRANSFORMER_MODEL_PATH, config.SCALER_PATH]):
        print(f"🚨 장기 예측 모델 또는 스케일러가 없습니다. train_transformer.py를 먼저 실행해주세요.")
        return

    scaler = joblib.load(config.SCALER_PATH)
    features = ['LAT', 'LON', '월', '요일', '피해면적_합계', 'IS_FIRE']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WildfireTransformer(
        n_features=len(features), 
        n_heads=len(features), 
        n_layers=2, 
        dropout=0.1, 
        pred_len=config.TRANSFORMER_PRED_LENGTH
    ).to(device)
    model.load_state_dict(torch.load(config.TRANSFORMER_MODEL_PATH, map_location=device))
    model.eval()

    print(f"\n⏳ 장기 예측을 위해 과거 {config.TRANSFORMER_SEQ_LENGTH}일치 데이터를 준비합니다.")
    day_map = {'월': 0, '화': 1, '수': 2, '목': 3, '금': 4, '토': 5, '일': 6}
    df['요일'] = df['발생일시_요일'].map(day_map)
    df_model_input = df[features].fillna(0).tail(config.TRANSFORMER_SEQ_LENGTH * len(locations_df))
    
    print(f"🔥 미래 {config.TRANSFORMER_PRED_LENGTH}일에 대한 장기 예측 수행 중...")
    all_preds_list = []
    for _, location_info in tqdm(locations_df.iterrows(), total=len(locations_df), desc="장기 예측"):
        loc_df = df_model_input.sample(config.TRANSFORMER_SEQ_LENGTH, replace=True).copy()
        loc_df['LAT'] = location_info['LAT']
        loc_df['LON'] = location_info['LON']
        
        input_data_scaled = scaler.transform(loc_df[features])
        input_tensor = torch.FloatTensor(input_data_scaled).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.sigmoid(output).squeeze().cpu().numpy()

        for i in range(config.TRANSFORMER_PRED_LENGTH):
            if probabilities[i] > 0.5:
                pred_info = location_info.copy()
                pred_info['fire_probability'] = probabilities[i]
                pred_info['prediction_day'] = i + 1
                all_preds_list.append(pred_info)

    if not all_preds_list:
        print(f"\n✅ 향후 {config.TRANSFORMER_PRED_LENGTH}일 동안 산불 위험이 예측된 지역이 없습니다.")
        return

    pred_df = pd.DataFrame(all_preds_list)
    print(f"\n✅ 향후 {config.TRANSFORMER_PRED_LENGTH}일 동안의 산불 위험 예측이 완료되었습니다. 결과를 JSON으로 저장합니다.")
    export_data_to_json(pred_df, 'transformer_predictions.json', extra_cols=['fire_probability', 'prediction_day'])
    open_dashboard()

def show_model_performance():
    """Opens the dashboard to show model performance."""
    print("\n📊 모델 성능 비교 대시보드를 엽니다.")
    open_dashboard()

def run_damage_prediction_mode():
    """Runs the damage prediction and opens the dashboard."""
    damage_prediction_path = "C:/Users/000/OneDrive/Desktop/fire_predictor_project/data/damage_predictions.json"
    if not os.path.exists(damage_prediction_path):
        print(f"🚨 피해 면적 예측 결과 파일({damage_prediction_path})이 없습니다.")
        print("지금 피해 면적 예측을 실행하시겠습니까? (y/n)")
        choice = input("> ")
        if choice.lower() == 'y':
            print("피해 면적 예측을 실행합니다...")
            os.system("python C:/Users/000/OneDrive/Desktop/fire_predictor_project/predict_damage_area.py")
        else:
            return
            
    print("\n📈 피해 면적 예측 결과를 대시보드에서 엽니다.")
    open_dashboard()

if __name__ == "__main__":
    if not os.path.exists(config.PREPROCESSED_DATA_PATH):
        print(f"🚨 데이터 파일({config.PREPROCESSED_DATA_PATH})이 없습니다. 데이터를 먼저 준비해주세요.")
        exit(1)

    print("- 산불 이력 데이터를 로딩합니다...")
    df = pd.read_csv(config.PREPROCESSED_DATA_PATH)
    locations_df = df[['LAT', 'LON', 'full_address']].drop_duplicates().reset_index(drop=True)
    df['IS_FIRE'] = (df['피해면적_합계'] > 0).astype(int)

    while True:
        print("\n========================================")
        print("* 산불 예측 및 조회 시스템 *")
        print("========================================")
        print("1: 과거 산불 이력 조회 및 대시보드 열기")
        print("2: 단기 산불 위험 예측 및 대시보드 열기")
        print("3: 장기 산불 위험 예측 및 대시보드 열기")
        print("4: 모델 성능 대시보드 열기")
        print("5: 피해 면적 예측 결과 보기")
        print("q: 종료")
        print("----------------------------------------")
        mode = input("원하는 작업의 번호를 입력하세요 > ")

        if mode == '1':
            run_historical_mode(df.copy())
        elif mode == '2':
            run_short_term_prediction_mode()
        elif mode == '3':
            run_long_term_prediction_mode(df.copy(), locations_df.copy())
        elif mode == '4':
            show_model_performance()
        elif mode == '5':
            run_damage_prediction_mode()
        elif mode.lower() == 'q':
            print("프로그램을 종료합니다.")
            break
        else:
            print("🚨 잘못된 입력입니다. 1, 2, 3, 4, 5, q 중에서 선택해주세요.")
