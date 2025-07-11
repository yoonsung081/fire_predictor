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
from sklearn.model_selection import TimeSeriesSplit, train_test_split
import lightgbm as lgb
from lightgbm import LGBMClassifier
from src.weather import get_weather
from tqdm import tqdm
import json
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import json
from sklearn.utils.class_weight import compute_class_weight
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


def create_sequences(data, target_data, seq_length, pred_length):
    """
    시계열 데이터를 시퀀스 형태로 변환합니다.
    X: (seq_length) 길이의 과거 데이터, y: (pred_length) 길이의 미래 데이터
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        x = data[i:(i + seq_length)]
        y = target_data[i + seq_length + pred_length - 1] # Target is the province ID of the event after the sequence
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def train_and_save_lightgbm_model(df, num_provinces, model_path="models/fire_predictor.joblib"):
    """
    데이터로 LightGBM 모델을 훈련하고 평가한 뒤, 전체 데이터로 재학습하여 저장합니다.
    """
    print("피처 엔지니어링 및 데이터 전처리 중...")
    
    datetime_cols = df[['발생일시_년', '발생일시_월', '발생일시_일']].copy()
    datetime_cols.columns = ['year', 'month', 'day']
    df['datetime'] = pd.to_datetime(datetime_cols, errors='coerce')
    df.dropna(subset=['datetime'], inplace=True)
    df = df.sort_values('datetime').reset_index(drop=True)

    features = ['LAT', 'LON', '월', '요일', '발생장소_시도', '발생장소_시군구', '발생원인_구분'] 
    target = '발생장소_시도_ID'

    # LightGBM이 범주형 특징을 직접 처리하도록 설정
    categorical_features = ['발생장소_시도', '발생장소_시군구', '발생원인_구분']

    for col in features:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    # LightGBM이 범주형 특징을 직접 처리하도록 설정
    for col in categorical_features:
        df[col] = df[col].astype('category')

    X = df[features]
    y = df[target]

    print("훈련/테스트 데이터 분할 중...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"훈련 데이터: {len(X_train)}개, 테스트 데이터: {len(X_test)}개")
    print(f"y_train 발생장소_시도_ID 분포:\n{y_train.value_counts()}")
    print(f"y_test 발생장소_시도_ID 분포:\n{y_test.value_counts()}")
    print(f"Full y (발생장소_시도_ID) distribution:\n{y.value_counts().to_string()}")

    print("모델 훈련 중 (LightGBM)...")
    lgbm = lgb.LGBMClassifier(objective='multiclass', num_class=num_provinces, random_state=42,
                      num_leaves=256, n_estimators=1000, max_depth=20, learning_rate=0.005, class_weight='balanced')
    lgbm.fit(X_train, y_train, categorical_feature=categorical_features)

    print("모델 평가 결과:")
    print("특징 중요도:")
    for i, importance in enumerate(lgbm.feature_importances_):
        print(f"  {X_train.columns[i]}: {importance:.4f}")
    y_pred = lgbm.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    accuracy = report['accuracy']
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']

    print(f"정확도: {accuracy:.4f}")
    print(f"정밀도 (Weighted Avg): {precision:.4f}")
    print(f"재현율 (Weighted Avg): {recall:.4f}")
    print(f"F1-score (Weighted Avg): {f1:.4f}")

    metrics = {
        "baseline": { # LightGBM 모델을 baseline으로 가정
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    }
    
    metrics_path = "static/metrics.json"
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"LightGBM model evaluation metrics saved: {metrics_path}")

    print("전체 데이터로 최종 모델 훈련 및 저장 중...")
    final_model = lgb.LGBMClassifier(objective='multiclass', num_class=num_provinces, random_state=42,
                             num_leaves=256, n_estimators=1000, max_depth=20, learning_rate=0.005, class_weight='balanced')
    final_model.fit(X, y, categorical_feature=categorical_features)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(final_model, model_path)
    print(f"Model saved: {model_path}")


def train_and_save_transformer(
    df, 
    num_provinces,
    model_path="models/transformer_predictor.pth", 
    scaler_path="models/transformer_scaler.joblib"
):
    """트랜스포머 모델을 훈련하고 저장합니다."""
    print("Transformer model training started...")

    features = ['LAT', 'LON', '월', '요일']
    df_model = df[features].fillna(0)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df_model)

    SEQ_LENGTH = 30
    PRED_LENGTH = 1 # Predicting one province at a time

    # Create sequences for X (features) and y (target province ID)
    xs, ys = [], []
    for i in range(len(data_scaled) - SEQ_LENGTH - PRED_LENGTH + 1):
        x = data_scaled[i:(i + SEQ_LENGTH)]
        # The target is the province ID of the event *after* the sequence
        y_val = df['발생장소_시도_ID'].iloc[i + SEQ_LENGTH + PRED_LENGTH - 1]
        xs.append(x)
        ys.append(y_val)
    
    X_sequences = np.array(xs)
    y_targets = np.array(ys)

    # 데이터 분할 전 y_targets의 클래스 분포 확인
    y_targets_series = pd.Series(y_targets)
    min_samples_per_class = y_targets_series.value_counts().min()

    # Perform train-test split on the sequences
    if min_samples_per_class < 2:
        print("경고: 일부 클래스에 샘플이 너무 적어 계층적 샘플링을 비활성화합니다.")
        X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_targets, test_size=0.2, random_state=42, shuffle=True)
    else:
        print("계층적 샘플링을 사용하여 데이터를 분할합니다.")
        X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_targets, test_size=0.2, random_state=42, stratify=y_targets, shuffle=True)
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train) # Target is now LongTensor for CrossEntropyLoss

    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False) # No need to shuffle test data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    # 클래스 가중치 계산 (훈련 데이터 불균형 해소)
    # y_train에 모든 클래스가 포함되지 않을 수 있으므로, 전체 클래스 수(num_provinces)를 기준으로 가중치 벡터를 생성합니다.
    # 이 경우, `compute_class_weight`는 y_train에 있는 클래스에 대한 가중치만 반환하므로, 전체 클래스에 대한 텐서로 확장해야 합니다.
    num_classes = num_provinces

    # y_train에 존재하는 고유 클래스와 해당 클래스에 대한 가중치를 계산합니다.
    unique_classes_in_train = np.unique(y_train)
    class_weights_for_present_classes = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes_in_train,
        y=y_train
    )

    # 전체 클래스에 대한 가중치 텐서를 생성하고, 기본값은 1.0으로 설정합니다.
    # 훈련 세트에 없는 클래스는 손실 계산에 사용되지 않으므로 가중치는 영향을 주지 않습니다.
    full_class_weights = np.ones(num_classes, dtype=np.float32)
    full_class_weights[unique_classes_in_train] = class_weights_for_present_classes

    class_weights_tensor = torch.tensor(full_class_weights, dtype=torch.float).to(device)

    model = WildfireTransformer(
        n_features=len(features), 
        n_heads=len(features),
        n_layers=2, 
        dropout=0.1, 
        pred_len=num_provinces # Output layer size is number of provinces
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor) # 가중치를 적용한 손실 함수
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    print(f"Training for {epochs} epochs...")
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
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Transformer model saved: {model_path}")
    print(f"Scaler saved: {scaler_path}")

    # Transformer 모델 평가
    print("Evaluating Transformer model...")
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_y in test_dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1) # Get the class with the highest probability
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    report_transformer = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)

    accuracy_transformer = report_transformer['accuracy']
    precision_transformer = report_transformer['weighted avg']['precision']
    recall_transformer = report_transformer['weighted avg']['recall']
    f1_transformer = report_transformer['weighted avg']['f1-score']

    print(f"정확도 (Transformer): {accuracy_transformer:.4f}")
    print(f"정밀도 (Weighted Avg): {precision_transformer:.4f}")
    print(f"재현율 (Weighted Avg): {recall_transformer:.4f}")
    print(f"F1-score (Weighted Avg): {f1_transformer:.4f}")

    metrics_path = "static/metrics.json"
    
    # 기존 metrics.json 파일이 있으면 로드, 없으면 새로 생성
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = {}

    metrics["transformer"] = {
        "accuracy": accuracy_transformer,
        "precision": precision_transformer,
        "recall": recall_transformer,
        "f1": f1_transformer
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Transformer model evaluation metrics saved: {metrics_path}")

if __name__ == "__main__":
    geojson_path = "C:/Users/000/OneDrive/Desktop/fire_predictor_project/data/true_fires.geojson"
    df = gpd.read_file(geojson_path)

    day_map = {'월': 0, '화': 1, '수': 2, '목': 3, '금': 4, '토': 5, '일': 6}
    df['요일'] = df['발생일시_요일'].map(day_map)

    # Map '발생장소_시도' to numerical labels
    province_to_id = {province: i for i, province in enumerate(df['발생장소_시도'].unique())}
    df['발생장소_시도_ID'] = df['발생장소_시도'].map(province_to_id)
    num_provinces = len(province_to_id)

    print(f"Unique provinces and their IDs: {province_to_id}")
    print(f"Number of unique provinces: {num_provinces}")

    print("--- Training LightGBM Model ---")
    train_and_save_lightgbm_model(df, num_provinces=num_provinces, model_path="models/lightgbm_province_predictor.joblib")

    print("--- Training Transformer Model ---")
    train_and_save_transformer(df, num_provinces=num_provinces, model_path="models/transformer_province_predictor.pth")