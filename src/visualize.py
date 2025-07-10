import os
import osmnx as ox
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from shapely.geometry import Polygon
import webbrowser
from tqdm import tqdm
from math import radians, sin, cos, sqrt, atan2

# ✅ 산 정상 정보 캐시 경로
PEAKS_CACHE = "data/peaks_cache.geojson"

# ✅ 범위 설정 (대한민국)
north, south, east, west = 43.0, 33.0, 132.0, 124.5
bbox_polygon = Polygon([
    (west, north), (east, north), (east, south),
    (west, south), (west, north)
])

# ✅ 산 위치 불러오기 (최초 1회만 API 호출)
if os.path.exists(PEAKS_CACHE):
    print("✅ 산 위치 캐시 불러오는 중...")
    gdf_peaks = gpd.read_file(PEAKS_CACHE)
else:
    print("🌐 OSM에서 산 위치 불러오는 중... (최초 1회)")
    gdf_peaks = ox.features_from_polygon(
        polygon=bbox_polygon,
        tags={"natural": "peak"}
    )
    gdf_peaks = gdf_peaks.to_crs(epsg=4326)
    os.makedirs("data", exist_ok=True)
    gdf_peaks.to_file(PEAKS_CACHE, driver="GeoJSON")
    print("✅ 캐시 저장 완료!")

# ✅ 거리 계산 함수
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

# ✅ 가장 가까운 산 정상 찾기
def find_nearest_peak(lat, lon):
    min_dist = float("inf")
    nearest = None
    for _, row in gdf_peaks.iterrows():
        try:
            peak_lat = row.geometry.y
            peak_lon = row.geometry.x
        except:
            continue
        dist = haversine(lat, lon, peak_lat, peak_lon)
        if dist < min_dist:
            min_dist = dist
            nearest = row
    if nearest is None:
        return lat, lon, ""
    return nearest.geometry.y, nearest.geometry.x, nearest.get("name", "")

# ✅ 지도 시각화 함수
def show_map_with_prediction(df_actual, df_predicted):
    print("🗺️ 지도 생성 중...")
    m = folium.Map(location=[37.5, 128.2], zoom_start=7)
    ca = MarkerCluster(name='과거 산불 위치').add_to(m)
    cp = MarkerCluster(name='예측 산불 위치').add_to(m)

    print("📍 과거 산불 위치 마커 추가 중...")
    for _, row in tqdm(df_actual.iterrows(), total=len(df_actual)):
        lat, lon, name = find_nearest_peak(row['LAT'], row['LON'])
        popup = f"📍 과거 산불<br>산: {name}<br>위치: {lat:.4f}, {lon:.4f}"
        folium.Marker(
            location=[lat, lon],
            popup=popup,
            icon=folium.Icon(color='red', icon='fire', prefix='fa')
        ).add_to(ca)

    print("🔥 예측 산불 위치 마커 추가 중...")
    for _, row in tqdm(df_predicted.iterrows(), total=len(df_predicted)):
        lat, lon, name = find_nearest_peak(row['LAT'], row['LON'])
        popup = f"🔥 예측 산불<br>산: {name}<br>위치: {lat:.4f}, {lon:.4f}"
        folium.Marker(
            location=[lat, lon],
            popup=popup,
            icon=folium.Icon(color='orange', icon='fire', prefix='fa')
        ).add_to(cp)

    folium.LayerControl().add_to(m)
    map_file = "map_result.html"
    m.save(map_file)
    print("✅ 지도 저장 완료. 웹 브라우저에서 여는 중...")
    webbrowser.open('file://' + os.path.realpath(map_file))
