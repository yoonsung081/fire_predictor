import geopandas as gpd
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os
import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c

def calculate_accuracy_by_distance(distance_km):
    if distance_km <= 0.2: return 90
    if distance_km <= 0.5: return 85
    if distance_km <= 1: return 80
    if distance_km <= 2: return 75
    if distance_km <= 5: return 70
    return max(0, 70 - (distance_km - 5) * 2)

def calculate_and_update_metrics(true_fires_path, predicted_fires_path, metrics_file_path, model_name="rule_based_prediction"):
    print(f"{model_name} 모델 지표 계산 및 업데이트 시작...")

    true_fires_gdf = gpd.read_file(true_fires_path)
    true_fires_gdf['is_fire'] = 1 

    predicted_fires_gdf = gpd.read_file(predicted_fires_path)

    # For overall metrics, we need to align true and predicted fires.
    # This part of the script assumes a direct merge or spatial join for overall metrics.
    # For individual marker accuracy, we'll handle it in add_accuracy_to_predictions.
    
    # This section for overall metrics might need refinement based on how true_fires and predicted_fires
    # are supposed to align for a single accuracy/precision/recall/f1 calculation.
    # For now, keeping the original logic for overall metrics.
    merged_df = pd.merge(true_fires_gdf, predicted_fires_gdf[['FIRE_RISK', 'geometry']], how='left', on='geometry')
    merged_df['FIRE_RISK'] = merged_df['FIRE_RISK'].fillna(0)

    y_true = merged_df['is_fire']
    y_pred = merged_df['FIRE_RISK']

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    print(f"계산된 지표: {metrics}")

    if os.path.exists(metrics_file_path):
        with open(metrics_file_path, 'r', encoding='utf-8') as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    all_metrics[model_name] = metrics

    with open(metrics_file_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    print(f"Metrics successfully updated in {metrics_file_path}.")

def add_accuracy_to_predictions(true_fires_path, predicted_fires_input_path, predicted_fires_output_path):
    print(f"Adding accuracy to predictions from {predicted_fires_input_path}...")
    true_fires_gdf = gpd.read_file(true_fires_path)
    predicted_fires_gdf = gpd.read_file(predicted_fires_input_path)

    predicted_fires_with_accuracy = []

    for idx, pred_fire in predicted_fires_gdf.iterrows():
        pred_lat, pred_lon = pred_fire.geometry.y, pred_fire.geometry.x
        min_distance = float('inf')
        
        for _, true_fire in true_fires_gdf.iterrows():
            true_lat, true_lon = true_fire.geometry.y, true_fire.geometry.x
            distance = haversine_distance(pred_lat, pred_lon, true_lat, true_lon)
            if distance < min_distance:
                min_distance = distance
        
        accuracy = calculate_accuracy_by_distance(min_distance)
        
        # Create a new feature with existing properties and the new accuracy_score
        new_properties = pred_fire.drop('geometry').to_dict()
        # Convert any Timestamp objects to string format
        for key, value in new_properties.items():
            if isinstance(value, pd.Timestamp):
                new_properties[key] = value.isoformat()
        new_properties['accuracy_score'] = accuracy
        new_properties['distance_to_closest_true_fire_km'] = min_distance # Optional: for debugging/info
        
        new_feature = {
            "type": "Feature",
            "properties": new_properties,
            "geometry": pred_fire.geometry.__geo_interface__
        }
        predicted_fires_with_accuracy.append(new_feature)

    output_geojson = {
        "type": "FeatureCollection",
        "features": predicted_fires_with_accuracy
    }

    with open(predicted_fires_output_path, 'w', encoding='utf-8') as f:
        json.dump(output_geojson, f, indent=2, ensure_ascii=False)
    print(f"Predictions with accuracy saved to {predicted_fires_output_path}")


if __name__ == "__main__":
    true_fires_path = "data/true_fires.geojson"
    metrics_file_path = "static/metrics.json"

    # Calculate and update overall metrics for satellite_rule_based model
    predicted_fires_path_satellite = "data/true_fires_with_weather_risk.geojson"
    calculate_and_update_metrics(true_fires_path, predicted_fires_path_satellite, metrics_file_path, model_name="satellite_rule_based")

    # Add accuracy to individual prediction files
    add_accuracy_to_predictions(true_fires_path, "data/predicted_weather.geojson", "data/predicted_weather_with_accuracy.geojson")
    add_accuracy_to_predictions(true_fires_path, "data/predicted_baseline.geojson", "data/predicted_baseline_with_accuracy.geojson")
    add_accuracy_to_predictions(true_fires_path, "data/true_fires_with_weather_risk.geojson", "data/predicted_satellite_with_accuracy.geojson")