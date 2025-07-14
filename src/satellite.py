"""
Google Earth Engine을 사용하여 위성 데이터를 가져오는 함수 모음
"""
import ee
import pandas as pd

def get_satellite_data(lat, lon, date):
    """
    특정 위치와 날짜의 Landsat 8 위성 데이터를 가져옵니다.
    NDVI와 지표면 온도(LST)를 계산하여 반환합니다.
    """
    try:
        ee.Initialize()
    except Exception as e:
        ee.Authenticate()
        ee.Initialize()

    point = ee.Geometry.Point(lon, lat)
    date = ee.Date(date)

    # Landsat 8 이미지 컬렉션 필터링
    image_collection = (
        ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
        .filterBounds(point)
        .filterDate(date.advance(-15, 'day'), date.advance(15, 'day')) # 날짜 주변 30일 검색
        .sort('CLOUD_COVER')
        .first()
    )

    if image_collection is None:
        return None, None

    # NDVI 계산 (B5: NIR, B4: Red)
    ndvi = image_collection.normalizedDifference(['B5', 'B4']).rename('NDVI')

    # 지표면 온도(LST) 계산 (B10: Thermal)
    lst = image_collection.select('B10').multiply(0.1).subtract(273.15).rename('LST') # 켈빈을 섭씨로 변환

    # 데이터 추출
    try:
        ndvi_val = ndvi.reduceRegion(ee.Reducer.mean(), point, 30).get('NDVI').getInfo()
        lst_val = lst.reduceRegion(ee.Reducer.mean(), point, 30).get('LST').getInfo()
        return ndvi_val, lst_val
    except Exception as e:
        # reduceRegion이 실패하는 경우 (예: 해당 지역에 데이터가 없음)
        return None, None

if __name__ == '__main__':
    # 테스트
    lat, lon = 36.3504, 127.3845 # 대전
    date = '2023-05-01'
    ndvi, lst = get_satellite_data(lat, lon, date)
    print(f"날짜: {date}")
    print(f"위치: (위도 {lat}, 경도 {lon})")
    print(f"NDVI: {ndvi}")
    print(f"지표면 온도 (LST): {lst}°C")
