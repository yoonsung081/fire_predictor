import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from src.load_data import load_all_csv
from src.geocoding import add_lat_lon_from_address
from src.model import add_fire_label, train_and_predict_baseline

# ✅ os import가 꼭 필요
cache_path = "data/with_coordinates.csv"
if os.path.exists(cache_path):
    df = pd.read_csv(cache_path)
else:
    df = load_all_csv()
    df = add_lat_lon_from_address(df)
    df.to_csv(cache_path, index=False)

df = add_fire_label(df)

df_pred = train_and_predict_baseline(df)

gdf = gpd.GeoDataFrame(
    df_pred,
    geometry=[Point(xy) for xy in zip(df_pred['LON'], df_pred['LAT'])],
    crs="EPSG:4326"
)
gdf.to_file("data/predicted_baseline.geojson", driver='GeoJSON')
print("✅ predicted_baseline.geojson 저장 완료")
