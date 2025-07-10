import folium
from folium.plugins import MarkerCluster
import webbrowser
import os

def show_comparison_map(center, df_true, df_pred_weather, df_pred_baseline):
    m = folium.Map(location=center, zoom_start=10)

    folium.TileLayer('OpenStreetMap', name='지도').add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='위성사진',
        overlay=False,
        control=True
    ).add_to(m)

    true_layer = MarkerCluster(name='실제 산불 위치').add_to(m)
    weather_layer = MarkerCluster(name='날씨 고려 예측').add_to(m)
    baseline_layer = MarkerCluster(name='날씨 미고려 예측').add_to(m)

    for _, row in df_true.iterrows():
        folium.Marker(
            [row['LAT'], row['LON']],
            popup="🔥 실제 산불",
            icon=folium.Icon(color='red', icon='fire', prefix='fa')
        ).add_to(true_layer)

    for _, row in df_pred_weather.iterrows():
        folium.Marker(
            [row['LAT'], row['LON']],
            popup="🌡️ 날씨 고려 예측",
            icon=folium.Icon(color='blue', icon='fire', prefix='fa')
        ).add_to(weather_layer)

    for _, row in df_pred_baseline.iterrows():
        folium.Marker(
            [row['LAT'], row['LON']],
            popup="⚡ 날씨 미고려 예측",
            icon=folium.Icon(color='orange', icon='fire', prefix='fa')
        ).add_to(baseline_layer)

    folium.LayerControl().add_to(m)
    m.save("map_result.html")
    webbrowser.open('file://' + os.path.realpath("map_result.html"))
