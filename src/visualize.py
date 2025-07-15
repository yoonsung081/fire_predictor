import folium
from folium.plugins import MarkerCluster
import webbrowser
import os

def show_fire_map(center, df, title, popup_col, color):
    m = folium.Map(location=center, zoom_start=10)

    folium.TileLayer('OpenStreetMap', name='지도').add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='위성사진',
        overlay=False,
        control=True
    ).add_to(m)

    marker_cluster = MarkerCluster(name=title).add_to(m)

    for _, row in df.iterrows():
        if row.get('display_prediction', True):
            popup_text = f"🔥 {row[popup_col]}"
            folium.Marker(
                [row['LAT'], row['LON']],
                popup=popup_text,
                icon=folium.Icon(color=color, icon='fire', prefix='fa')
            ).add_to(marker_cluster)

    folium.LayerControl().add_to(m)
    m.save("map_result.html")
    webbrowser.open('file://' + os.path.realpath("map_result.html"))

def show_long_term_prediction_map(center, pred_dfs):
    m = folium.Map(location=center, zoom_start=7)
    folium.TileLayer('OpenStreetMap', name='지도').add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='위성사진',
        overlay=False,
        control=True
    ).add_to(m)

    for day, df in pred_dfs.items():
        layer = MarkerCluster(name=f'{day} 예측').add_to(m)
        for _, row in df.iterrows():
            if row.get('display_prediction', True): # display_prediction이 True인 경우에만 마커 표시
                folium.Marker(
                    [row['LAT'], row['LON']],
                    popup=f"🔥 예측 확률: {row['fire_probability']:.2f}",
                    icon=folium.Icon(color='purple', icon='fire', prefix='fa')
                ).add_to(layer)

    folium.LayerControl().add_to(m)
    m.save("long_term_prediction_map.html")
    webbrowser.open('file://' + os.path.realpath("long_term_prediction_map.html"))
