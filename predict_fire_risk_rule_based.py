import geopandas as gpd
import pandas as pd
import os
import json
from tqdm import tqdm
from src.weather import get_weather
import numpy as np

def predict_fire_risk_rule_based(input_geojson_path, output_geojson_path,
                                 temp_threshold=25, precip_threshold=0.1, wind_threshold=5):
    print("✨ 규칙 기반 산불 위험 예측 시작...")

    # GeoJSON 파일 로드
    df = gpd.read_file(input_geojson_path)

    # 날짜 컬럼 생성 (가정: predicted_weather.geojson에 날짜 정보가 없으므로, 임의의 날짜를 사용하거나,
    # 실제 예측하고자 하는 날짜를 외부에서 주입해야 합니다. 여기서는 2024년 1월 1일로 가정합니다.)
    # 실제 사용 시에는 예측하고자 하는 날짜를 정확히 설정해야 합니다.
    df['datetime'] = pd.to_datetime(df[['발생일시_년', '발생일시_월', '발생일시_일']].astype(str).agg('-'.join, axis=1)) 

    # 날씨 캐시 로드
    weather_cache_path = "data/weather_cache.json"
    if os.path.exists(weather_cache_path):
        with open(weather_cache_path, 'r') as f:
            weather_cache = json.load(f)
    else:
        weather_cache = {}

    # 날씨 데이터 가져오기 및 통합
    print("Fetching weather data for rule-based prediction...")
    weather_data_list = []
    new_fetches_count = 0
    max_new_fetches = 100 # 100개로 제한 (조절 가능)

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        lat = row['LAT']
        lon = row['LON']
        date_str = row['datetime'].strftime('%Y-%m-%d')
        
        cache_key = f"{lat},{lon},{date_str}"
        if cache_key in weather_cache:
            weather_info = weather_cache[cache_key]
        else:
            if new_fetches_count < max_new_fetches:
                weather_info = get_weather(lat, lon, date_str)
                if weather_info:
                    weather_cache[cache_key] = weather_info
                    new_fetches_count += 1
                else:
                    weather_info = {'temp_max': np.nan, 'temp_min': np.nan, 'precip': np.nan, 'wind': np.nan}
            else:
                # 새로운 fetch 제한에 도달하면 NaN으로 채움
                weather_info = {'temp_max': np.nan, 'temp_min': np.nan, 'precip': np.nan, 'wind': np.nan}
        weather_data_list.append(weather_info)

    weather_df = pd.DataFrame(weather_data_list)
    df = pd.concat([df, weather_df], axis=1)

    # 업데이트된 날씨 캐시 저장
    with open(weather_cache_path, 'w') as f:
        json.dump(weather_cache, f, indent=4)
    print("Weather data fetching complete and cache updated.")

    # 결측치 처리 (날씨 데이터가 없는 경우)
    for col in ['temp_max', 'temp_min', 'precip', 'wind']:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # 규칙 기반 예측
    print("Applying rule-based prediction...")
    df['FIRE_RISK'] = 0 # 기본값은 위험 없음

    # 산불 위험 조건: 고온, 건조, 강풍
    # temp_max >= temp_threshold
    # precip < precip_threshold
    # wind >= wind_threshold
    df.loc[(df['temp_max'] >= temp_threshold) & 
           (df['precip'] < precip_threshold) & 
           (df['wind'] >= wind_threshold), 'FIRE_RISK'] = 1

    # 결과 저장
    os.makedirs(os.path.dirname(output_geojson_path), exist_ok=True)
    df.to_file(output_geojson_path, driver='GeoJSON')
    print(f"규칙 기반 산불 위험 예측 결과 저장: {output_geojson_path}")

if __name__ == "__main__":
    input_path = "data/true_fires.geojson"
    output_path = "data/true_fires_with_weather_risk.geojson"

    # 규칙 설정 (여기서 값을 조정하여 다양한 시나리오 테스트 가능)
    # 예시: 최고 기온 25도 이상, 강수량 0.1mm 미만, 풍속 5m/s 이상
    predict_fire_risk_rule_based(input_path, output_path,
                                 temp_threshold=30, precip_threshold=0.05, wind_threshold=7)
