import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 로드
try:
    df = pd.read_csv("C:\\Users\\000\\OneDrive\\Desktop\\fire_predictor_project\\data\\산림청_산불상황관제시스템 산불통계데이터_20241016.csv", encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv("C:\\Users\\000\\OneDrive\\Desktop\\fire_predictor_project\\data\\산림청_산불상황관제시스템 산불통계데이터_20241016.csv", encoding='utf-8-sig')

# 데이터 전처리
df = df.drop(['진화종료시간_년', '진화종료시간_월', '진화종료시간_일', '진화종료시간_시간', '발생원인_기타'], axis=1)
df = df.dropna(subset=['피해면적_합계'])

# 주소 합치기
df['full_address'] = df['발생장소_시도'].fillna('') + ' ' + df['발생장소_시군구'].fillna('') + ' ' + df['발생장소_읍면'].fillna('') + ' ' + df['발생장소_동리'].fillna('')

# 위도, 경도 추가 (실제로는 지오코딩 필요)
from src.geocoding import add_lat_lon_from_address
print("주소로부터 위도, 경도 정보를 가져옵니다. 이 과정은 시간이 오래 걸릴 수 있습니다...")
df = add_lat_lon_from_address(df)
df = df.dropna(subset=['LAT', 'LON'])
print("위도, 경도 정보 추가 완료.")

categorical_features = ['발생일시_요일', '발생장소_관서', '발생장소_시도', '발생장소_시군구', '발생장소_읍면', '발생장소_동리', '발생원인_구분', '발생원인_세부원인']
for col in categorical_features:
    df[col] = df[col].astype('category')

features = ['발생일시_년', '발생일시_월', '발생일시_일', 'LAT', 'LON'] + categorical_features
X = df[features]
y = df['피해면적_합계']

# 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM 모델 학습
lgb_train = lgb.Dataset(X_train, y_train, feature_name=features, categorical_feature=categorical_features)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, feature_name=features, categorical_feature=categorical_features)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

print("피해면적 예측 모델 학습을 시작합니다...")
model = lgb.train(params,
                  lgb_train,
                  num_boost_round=1000,
                  valid_sets=[lgb_train, lgb_eval],
                  callbacks=[lgb.early_stopping(100)])

# 모델 평가
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"모델 학습 완료. RMSE: {rmse:.4f}, R2 Score: {r2:.4f}")

# 모델 저장
joblib.dump(model, "C:\\Users\\000\\OneDrive\\Desktop\\fire_predictor_project\\models\\damage_regression_model.joblib")
print("학습된 모델을 'models/damage_regression_model.joblib'에 저장했습니다.")