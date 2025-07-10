from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

def add_fire_label(df):
    if 'IS_FIRE' not in df.columns:
        print("🧠 IS_FIRE 라벨 생성 중...")
        df['IS_FIRE'] = df['피해면적_합계'].apply(
            lambda x: 1 if pd.notnull(x) and x > 0 else 0
        )
    return df

def train_and_predict(df):
    # 기상 데이터가 없으므로 날짜 기반 피처 생성
    df['월'] = df['발생일시_월']
    df['요일'] = df['발생일시_요일'].astype('category').cat.codes  # 문자열 요일 -> 숫자화

    df = df.dropna(subset=['LAT', 'LON', '월', '요일'])

    X = df[['월', '요일']]
    y = df['IS_FIRE']
    lat = df['LAT']
    lon = df['LON']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    lat_test = lat.loc[X_test.index]
    lon_test = lon.loc[X_test.index]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    result = pd.DataFrame({
        'LAT': lat_test.values,
        'LON': lon_test.values,
        'PREDICTED_FIRE': y_pred
    })
    return result
