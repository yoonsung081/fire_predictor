from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from src.weather import get_weather

def add_fire_label(df):
    if 'IS_FIRE' not in df.columns:
        df['IS_FIRE'] = df['피해면적_합계'].apply(lambda x: 1 if pd.notnull(x) and x > 0 else 0)
    return df

def add_weather_features(df):
    features = []
    for _, row in df.iterrows():
        date = f"{int(row['발생일시_년']):04d}-{int(row['발생일시_월']):02d}-{int(row['발생일시_일']):02d}"
        lat, lon = row['LAT'], row['LON']
        weather = get_weather(lat, lon, date)
        if weather:
            features.append(weather)
        else:
            features.append({"temp_max": 0, "temp_min": 0, "precip": 0, "wind": 0})
    df_weather = pd.DataFrame(features)
    df = pd.concat([df.reset_index(drop=True), df_weather.reset_index(drop=True)], axis=1)
    return df

def train_and_predict_weather(df):
    df = df.dropna(subset=['LAT', 'LON'])
    df = add_weather_features(df)
    X = df[['temp_max', 'temp_min', 'precip', 'wind']]
    y = df['IS_FIRE']
    X_train, X_test, y_train, y_test, lat_test, lon_test = train_test_split(
        X, y, df['LAT'], df['LON'], test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    df_pred = pd.DataFrame({
        'LAT': lat_test,
        'LON': lon_test,
        'PREDICTED_FIRE': y_pred
    })
    return df_pred[df_pred['PREDICTED_FIRE'] == 1]

def train_and_predict_baseline(df):
    df = df.dropna(subset=['LAT', 'LON'])
    X = df[['LAT', 'LON']]  # 날씨 없이 baseline feature
    y = df['IS_FIRE']

    # lat_test, lon_test 분리 안 하고 단순히 index를 기준으로 join
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    df_test = X_test.copy()
    df_test['PREDICTED_FIRE'] = y_pred
    return df_test[df_test['PREDICTED_FIRE'] == 1]
