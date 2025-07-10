import folium
from folium.plugins import MarkerCluster
import webbrowser
import os

def show_map_with_prediction(df_actual, df_pred):
    m = folium.Map(location=[37.5, 128.2], zoom_start=7)

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

    for _, row in df_actual.iterrows():
        folium.Marker(
            [row['LAT'], row['LON']],
            popup=f"📍 과거 산불",
            icon=folium.Icon(color='red', icon='fire', prefix='fa')
        ).add_to(ca)

    for _, row in df_pred.iterrows():
        folium.Marker(
            [row['LAT'], row['LON']],
            popup=f"🔥 예측 산불",
            icon=folium.Icon(color='orange', icon='fire', prefix='fa')
        ).add_to(cp)

    folium.LayerControl().add_to(m)
    m.save("map_result.html")
    webbrowser.open('file://' + os.path.realpath("map_result.html"))
