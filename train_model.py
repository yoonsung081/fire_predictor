
"""
모델 훈련 및 저장 스크립트
LightGBM 또는 Transformer 모델을 훈련하고 저장합니다.
"""
import geopandas as gpd
import pandas as pd
import os
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

from src.load_data import load_all_csv
from src.geocoding import add_lat_lon_from_address
from src.model import add_fire_label

class WildfireTransformer(nn.Module):
    """
    간단한 인코더 기반 트랜스포머 모델.
    과거 시퀀스를 입력받아 미래의 산불 발생 확률을 예측합니다.
    """
    def __init__(self, n_features, n_heads, n_layers, dropout, pred_len):
        super().__init__()
        self.n_features = n_features
        self.pred_len = pred_len

        # 인코더 레이어 정의
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_features, 
            nhead=n_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 최종 예측을 위한 선형 레이어
        self.output_layer = nn.Linear(n_features, pred_len)

    def forward(self, src):
        # src: (batch_size, seq_len, n_features)
        encoded = self.transformer_encoder(src)
        # 마지막 타임스텝의 출력을 사용하여 미래를 예측
        output = self.output_layer(encoded[:, -1, :]) # (batch_size, pred_len)
        return output


def create_sequences(data, seq_length, pred_length):
    """
    시계열 데이터를 시퀀스 형태로 변환합니다.
    X: (seq_length) 길이의 과거 데이터, y: (pred_length) 길이의 미래 데이터
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        x = data[i:(i + seq_length)]
        y = data[(i + seq_length):(i + seq_length + pred_length), -1] # 타겟은 마지막 'IS_FIRE' 컬럼
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def train_and_save_lightgbm_model(df, model_path="models/fire_predictor.joblib"):
    """
    데이터로 LightGBM 모델을 훈련하고 평가한 뒤, 전체 데이터로 재학습하여 저장합니다.
    """
    print("✨ 피처 엔지니어링 및 데이터 전처리 중...")
    
    datetime_cols = df[['발생일시_년', '발생일시_월', '발생일시_일']].copy()
    datetime_cols.columns = ['year', 'month', 'day']
    df['datetime'] = pd.to_datetime(datetime_cols, errors='coerce')
    df.dropna(subset=['datetime'], inplace=True)
    df = df.sort_values('datetime').reset_index(drop=True)

    features = ['LAT', 'LON', '월', '요일', '피해면적_합계'] 
    target = 'IS_FIRE'

    for col in features:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    X = df[features]
    y = df[target]

    print("⏳ 시계열 교차검증으로 모델 성능 평가 중...")
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    print(f"훈련 데이터: {len(X_train)}개, 테스트 데이터: {len(X_test)}개")

    print("🚀 모델 훈련 중 (LightGBM)...")
    lgbm = LGBMClassifier(objective='binary', is_unbalance=True, random_state=42)
    lgbm.fit(X_train, y_train)

    print("📊 모델 평가 결과:")
    y_pred = lgbm.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['No Fire', 'Fire'], zero_division=0, labels=[0, 1]))

    print("💾 전체 데이터로 최종 모델 훈련 및 저장 중...")
    final_model = LGBMClassifier(objective='binary', is_unbalance=True, random_state=42)
    final_model.fit(X, y)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(final_model, model_path)
    print(f"✅ 모델 저장 완료: {model_path}")


def train_and_save_transformer(
    df, 
    model_path="models/transformer_predictor.pth", 
    scaler_path="models/transformer_scaler.joblib"
):
    """트랜스포머 모델을 훈련하고 저장합니다."""
    print("🤖 트랜스포머 모델 훈련 시작...")

    features = ['LAT', 'LON', '월', '요일', '피해면적_합계', 'IS_FIRE']
    datetime_cols = df[['발생일시_년', '발생일시_월', '발생일시_일']].copy()
    datetime_cols.columns = ['year', 'month', 'day']
    df['datetime'] = pd.to_datetime(datetime_cols, errors='coerce')
    df.dropna(subset=['datetime'], inplace=True)
    df = df.sort_values('datetime').reset_index(drop=True)
    
    df_model = df[features].fillna(0)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df_model)

    SEQ_LENGTH = 30
    PRED_LENGTH = 7
    X, y = create_sequences(data_scaled, SEQ_LENGTH, PRED_LENGTH)

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    model = WildfireTransformer(
        n_features=len(features), 
        n_heads=len(features),
        n_layers=2, 
        dropout=0.1, 
        pred_len=PRED_LENGTH
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    print(f"🚀 {epochs} 에포크 동안 훈련 진행...")
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    print(f"✅ 트랜스포머 모델 저장 완료: {model_path}")
    print(f"✅ 스케일러 저장 완료: {scaler_path}")

if __name__ == "__main__":
    # 모델 선택
    MODEL_TYPE = "lightgbm" # "lightgbm" 또는 "transformer"

    geojson_path = "data/true_fires.geojson"
    df = gpd.read_file(geojson_path)


    df['IS_FIRE'] = (df['피해면적_합계'] > 0).astype(int)
    
    day_map = {'월': 0, '화': 1, '수': 2, '목': 3, '금': 4, '토': 5, '일': 6}
    df['요일'] = df['발생일시_요일'].map(day_map)

    if MODEL_TYPE == "lightgbm":
        train_and_save_lightgbm_model(df)
    elif MODEL_TYPE == "transformer":
        train_and_save_transformer(df)
    else:
        print(f"🚨 잘못된 모델 타입입니다: {MODEL_TYPE}")
