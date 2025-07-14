"""
모델 훈련 및 저장 스크립트
LightGBM 또는 Transformer 모델을 훈련하고 저장합니다.
위성 데이터 사용 여부에 따라 모델을 분리하여 훈련하고 성능을 비교합니다.
"""
import pandas as pd
import os
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import lightgbm as lgb

import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from src.satellite import get_satellite_data

# tqdm pandas integration


class WildfireTransformer(nn.Module):
    """
    간단한 인코더 기반 트랜스포머 모델.
    """
    def __init__(self, n_features, n_heads, n_layers, dropout, pred_len=1):
        super().__init__()
        self.n_features = n_features
        self.pred_len = pred_len
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_features, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_layer = nn.Linear(n_features, pred_len)

    def forward(self, src):
        encoded = self.transformer_encoder(src)
        output = self.output_layer(encoded[:, -1, :])
        return output

def create_sequences(data, target_data, seq_length):
    """
    시계열 데이터를 시퀀스 형태로 변환합니다.
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = target_data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_and_save_lightgbm_model(df, model_path="models/fire_predictor.joblib", use_satellite=False):
    """
    데이터로 LightGBM 이진 분류 모델을 훈련하고 저장합니다.
    """
    model_type = "LGBM_with_satellite" if use_satellite else "LGBM_baseline"
    print(f"--- {model_type} 모델 훈련 시작 ---")

    features = ['LAT', 'LON', '월', '요일']
    if use_satellite:
        features.extend(['NDVI', 'LST'])
    
    target = 'IS_FIRE'

    for col in features:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"[{model_type}] 훈련 데이터: {len(X_train)}개, 테스트 데이터: {len(X_test)}개")
    
    print(f"[{model_type}] 모델 훈련 중...")
    lgbm = lgb.LGBMClassifier(objective='binary', random_state=42, n_estimators=500, learning_rate=0.05, num_leaves=31, is_unbalance=True)
    lgbm.fit(X_train, y_train)

    print(f"[{model_type}] 모델 평가 결과:")
    y_pred = lgbm.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"정확도: {accuracy:.4f}, 정밀도: {precision:.4f}, 재현율: {recall:.4f}, F1-score: {f1:.4f}")

    metrics_path = "static/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = {}
    
    metric_key = "baseline_with_satellite" if use_satellite else "baseline"
    metrics[metric_key] = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[{model_type}] 모델 평가 지표 저장 완료: {metrics_path}")

    print(f"[{model_type}] 전체 데이터로 최종 모델 훈련 및 저장 중...")
    final_model = lgb.LGBMClassifier(objective='binary', random_state=42, n_estimators=500, learning_rate=0.05, num_leaves=31, is_unbalance=True)
    final_model.fit(X, y)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(final_model, model_path)
    print(f"모델 저장 완료: {model_path}")

def train_and_save_transformer(df, model_path="models/transformer_predictor.pth", scaler_path="models/transformer_scaler.joblib", use_satellite=False):
    """트랜스포머 이진 분류 모델을 훈련하고 저장합니다."""
    model_type = "Transformer_with_satellite" if use_satellite else "Transformer_baseline"
    print(f"--- {model_type} 모델 훈련 시작 ---")

    features = ['LAT', 'LON', '월', '요일', 'IS_FIRE']
    if use_satellite:
        features.insert(-1, 'NDVI')
        features.insert(-1, 'LST')

    target = 'IS_FIRE'
    
    df_model = df[features].fillna(0)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df_model)

    SEQ_LENGTH = 30
    
    X_sequences, y_sequences = create_sequences(data_scaled, data_scaled[:, -1], SEQ_LENGTH)

    if len(X_sequences) == 0:
        print(f"[{model_type}] 시퀀스 생성 후 데이터가 없어 훈련을 건너뜁니다.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42, stratify=y_sequences if np.sum(y_sequences) > 1 else None)
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    model = WildfireTransformer(n_features=data_scaled.shape[1], n_heads=data_scaled.shape[1], n_layers=2, dropout=0.1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    print(f"[{model_type}] {epochs} 에포크 동안 훈련 진행...")
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f'에포크 [{epoch+1}/{epochs}], 손실: {loss.item():.4f}')

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    print(f"모델 저장 완료: {model_path}, 스케일러 저장 완료: {scaler_path}")

    print(f"[{model_type}] 모델 평가 중...")
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_dataloader:
            outputs = model(batch_X.to(device))
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(batch_y.cpu().numpy().flatten())

    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    print(f"정확도: {accuracy:.4f}, 정밀도: {precision:.4f}, 재현율: {recall:.4f}, F1-score: {f1:.4f}")

    metrics_path = "static/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = {}
        
    metric_key = "transformer_with_satellite" if use_satellite else "transformer"
    metrics[metric_key] = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[{model_type}] 모델 평가 지표 저장 완료: {metrics_path}")


if __name__ == "__main__":
    data_path = "data/with_coordinates.csv"
    if not os.path.exists(data_path):
        print(f"데이터 파일({data_path})이 없습니다. app.py를 실행하여 데이터를 생성해주세요.")
        exit(1)
        
    print("데이터 로딩 중...")
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    df.columns = df.columns.str.strip() # 컬럼 이름의 공백 제거

    # 첫 번째 컬럼 이름에서 '발생일시_년'을 포함하는 실제 컬럼 이름을 찾아서 수정
    actual_first_col = df.columns[0]
    if '발생일시_년' in actual_first_col:
        # '발생일시_년'이 시작하는 인덱스를 찾아서 그 부분부터 컬럼 이름으로 사용
        start_index = actual_first_col.find('발생일시_년')
        corrected_first_col = actual_first_col[start_index:]
        df.rename(columns={actual_first_col: corrected_first_col}, inplace=True)
    
    df['IS_FIRE'] = (df['피해면적_합계'] > 0).astype(int)
    day_map = {'월': 0, '화': 1, '수': 2, '목': 3, '금': 4, '토': 5, '일': 6}
    df['요일'] = df['발생일시_요일'].map(day_map)
    df['datetime'] = pd.to_datetime(df['발생일시_년'].astype(str) + '-' + df['발생일시_월'].astype(str) + '-' + df['발생일시_일'].astype(str), errors='coerce')
    df['월'] = df['datetime'].dt.month # '월' 컬럼 추가
    df.dropna(subset=['datetime'], inplace=True)
    df = df.sort_values('datetime').reset_index(drop=True)

    # --- 베이스라인 모델 훈련 ---
    print("\n--- 베이스라인 모델 훈련 (위성 데이터 미사용) ---")
    train_and_save_lightgbm_model(df.copy(), model_path="models/fire_predictor.joblib", use_satellite=False)
    train_and_save_transformer(df.copy(), model_path="models/transformer_predictor.pth", scaler_path="models/transformer_scaler.joblib", use_satellite=False)

    # --- 위성 데이터 추가 ---
    print("\n--- 위성 데이터 추가 중 (시간이 소요될 수 있습니다) ---")
    cache_path = "data/satellite_cache.csv"
    if os.path.exists(cache_path):
        print("캐시된 위성 데이터 로딩...")
        sat_df = pd.read_csv(cache_path)
        sat_df['datetime'] = pd.to_datetime(sat_df['datetime'])
    else:
        print("새로운 위성 데이터 조회...")
        unique_requests = df[['LAT', 'LON', 'datetime']].drop_duplicates().copy()
        
        results = unique_requests.apply(
            lambda row: get_satellite_data(row['LAT'], row['LON'], row['datetime'].strftime('%Y-%m-%d')),
            axis=1,
            result_type='expand'
        )
        results.columns = ['NDVI', 'LST']
        
        sat_df = pd.concat([unique_requests.reset_index(drop=True), results], axis=1)
        sat_df.to_csv(cache_path, index=False)

    df_with_sat = pd.merge(df, sat_df, on=['LAT', 'LON', 'datetime'], how='left')
    df_with_sat['NDVI'].fillna(0, inplace=True)
    df_with_sat['LST'].fillna(df_with_sat['LST'].mean(), inplace=True)
    print("위성 데이터 추가 완료.")

    # --- 위성 데이터를 사용한 모델 훈련 ---
    print("\n--- 위성 데이터를 사용한 모델 훈련 ---")
    train_and_save_lightgbm_model(df_with_sat.copy(), model_path="models/fire_predictor_with_satellite.joblib", use_satellite=True)
    train_and_save_transformer(df_with_sat.copy(), model_path="models/transformer_predictor_with_satellite.pth", scaler_path="models/transformer_scaler_with_satellite.joblib", use_satellite=True)