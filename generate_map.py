import pandas as pd
import folium
from folium.plugins import MarkerCluster
import webbrowser
import os

print("📂 CSV 파일 불러오는 중...")
df_actual = pd.read_csv("data/fixed_actual.csv")
df_pred = pd.read_csv("data/fixed_predict.csv")

m = folium.Map(location=[37.5, 128.2], zoom_start=7)

# ✅ 지도 스타일 토글 추가
folium.TileLayer('OpenStreetMap', name='지도').add_to(m)
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri',
    name='위성사진',
    overlay=False,
    control=True
).add_to(m)

ca = MarkerCluster(name='과거 산불 위치').add_to(m)
cp = MarkerCluster(name='예측 산불 위치').add_to(m)

print("📍 과거 산불 마커 그리는 중...")
for _, row in df_actual.iterrows():
    folium.Marker(
        [row['LAT'], row['LON']],
        popup=f"📍 과거 산불",
        icon=folium.Icon(color='red', icon='fire', prefix='fa')
    ).add_to(ca)

print("🔥 예측 산불 마커 그리는 중...")
for _, row in df_pred.iterrows():
    folium.Marker(
        [row['LAT'], row['LON']],
        popup=f"🔥 예측 산불",
        icon=folium.Icon(color='orange', icon='fire', prefix='fa')
    ).add_to(cp)

folium.LayerControl().add_to(m)

map_file = "map_result.html"
m.save(map_file)
print("✅ map_result.html 저장 완료")
webbrowser.open('file://' + os.path.realpath(map_file))
