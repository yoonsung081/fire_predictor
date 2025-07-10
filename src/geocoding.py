from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import pandas as pd

def add_lat_lon_from_address(df):
    if 'LAT' in df.columns and 'LON' in df.columns:
        print("위도/경도 열이 이미 존재합니다. 변환 생략.")
        return df

    print("주소 기반 위도/경도 변환 시작...")
    geolocator = Nominatim(user_agent="fire_geo_locator")
    geocode = RateLimiter(lambda addr: geolocator.geocode(addr, timeout=5),
                          min_delay_seconds=1,
                          max_retries=2,
                          error_wait_seconds=2)

    # ✅ 주소 열 조합 (시도 + 시군구 + 읍면)
    df['full_address'] = (
        "대한민국 "
        + df['발생장소_시도'].astype(str) + " "
        + df['발생장소_시군구'].astype(str) + " "
        + df['발생장소_읍면'].astype(str)
    )

    lat_list, lon_list = [], []

    for i, addr in enumerate(df['full_address']):
        try:
            location = geocode(addr)
            if location:
                lat_list.append(location.latitude)
                lon_list.append(location.longitude)
            else:
                lat_list.append(None)
                lon_list.append(None)
        except:
            lat_list.append(None)
            lon_list.append(None)

        if i % 10 == 0 or i == len(df) - 1:
            print(f"진행 중... {i+1}/{len(df)}개 처리 완료")

    df['LAT'] = lat_list
    df['LON'] = lon_list
    print("위도/경도 변환 완료")
    return df

def filter_to_korea(df):
    return df[
        (df['LAT'] >= 33) & (df['LAT'] <= 39) &
        (df['LON'] >= 124) & (df['LON'] <= 132)
    ]
