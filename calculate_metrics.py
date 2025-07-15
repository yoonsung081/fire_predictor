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
    if distance_km <= 0.05:
        return 90
    if distance_km <= 1.0:
        steps = np.floor((distance_km - 0.001) / 0.05)
        accuracy = 90 - steps
        return max(80, int(accuracy))
    if distance_km <= 2:
        return 75
    if distance_km <= 5:
        return 70
    else: # distance_km > 5
        return 0 # 5km를 초과하는 거리에 대해서는 정확도를 0으로 설정

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

def filter_predictions_for_display(predicted_fires_gdf, true_fires_gdf, limit=3, radius_km=5.0):
    """
    Filters predicted fires to display only a limited number (default 3) closest to each true fire.
    Sets 'display_prediction' column to True for predictions that should be displayed.
    """
    predicted_fires_gdf['display_prediction'] = False
    predicted_fires_gdf['accuracy_score'] = 0
    predicted_fires_gdf['distance_to_closest_true_fire_km'] = np.inf

    predictions_to_display_indices = set()

    for _, true_fire in true_fires_gdf.iterrows():
        true_lat, true_lon = true_fire.geometry.y, true_fire.geometry.x
        true_date = true_fire.get('date')

        if true_date:
            predicted_fires_on_same_date = predicted_fires_gdf[predicted_fires_gdf['date'] == true_date]
        else:
            predicted_fires_on_same_date = predicted_fires_gdf

        nearby_predictions = []
        for pred_idx, pred_fire in predicted_fires_on_same_date.iterrows():
            pred_lat, pred_lon = pred_fire.geometry.y, pred_fire.geometry.x
            distance = haversine_distance(true_lat, true_lon, pred_lat, pred_lon)

            if distance <= radius_km:
                nearby_predictions.append((distance, pred_idx))

        nearby_predictions.sort(key=lambda x: x[0])
        for i, (distance, pred_idx) in enumerate(nearby_predictions):
            if i < limit:
                predictions_to_display_indices.add(pred_idx)
                predicted_fires_gdf.loc[pred_idx, 'accuracy_score'] = calculate_accuracy_by_distance(distance)
                predicted_fires_gdf.loc[pred_idx, 'distance_to_closest_true_fire_km'] = distance
                predicted_fires_gdf.loc[pred_idx, 'display_prediction'] = True

    return predicted_fires_gdf

def add_accuracy_to_predictions(true_fires_path, predicted_fires_input_path, predicted_fires_output_path, start_date=None, end_date=None):
    print(f"Adding accuracy to predictions from {predicted_fires_input_path}...")
    true_fires_gdf = gpd.read_file(true_fires_path)
    predicted_fires_gdf = gpd.read_file(predicted_fires_input_path)

    # Ensure 'date' columns are standardized to string 'YYYY-MM-DD' format
    if 'date' in true_fires_gdf.columns:
        true_fires_gdf['date'] = pd.to_datetime(true_fires_gdf['date']).dt.strftime('%Y-%m-%d')
    if 'date' in predicted_fires_gdf.columns:
        predicted_fires_gdf['date'] = pd.to_datetime(predicted_fires_gdf['date']).dt.strftime('%Y-%m-%d')

    # Apply date filtering if start_date and end_date are provided
    if start_date and end_date:
        start_date_dt = pd.to_datetime(start_date).date()
        end_date_dt = pd.to_datetime(end_date).date()

        if 'date' in true_fires_gdf.columns:
            true_fires_gdf['datetime'] = pd.to_datetime(true_fires_gdf['date'])
            true_fires_gdf = true_fires_gdf[(true_fires_gdf['datetime'].dt.date >= start_date_dt) &
                                            (true_fires_gdf['datetime'].dt.date <= end_date_dt)]
            true_fires_gdf = true_fires_gdf.drop(columns=['datetime']) # Remove temporary column

        if 'date' in predicted_fires_gdf.columns:
            predicted_fires_gdf['datetime'] = pd.to_datetime(predicted_fires_gdf['date'])
            predicted_fires_gdf = predicted_fires_gdf[(predicted_fires_gdf['datetime'].dt.date >= start_date_dt) &
                                                      (predicted_fires_gdf['datetime'].dt.date <= end_date_dt)]
            predicted_fires_gdf = predicted_fires_gdf.drop(columns=['datetime']) # Remove temporary column

    # Filter predictions for display
    predicted_fires_gdf = filter_predictions_for_display(predicted_fires_gdf, true_fires_gdf)

    # Convert GeoDataFrame to a list of features for JSON output
    predicted_fires_with_accuracy = []
    for idx, row in predicted_fires_gdf.iterrows():
        new_properties = row.drop(['geometry', 'display_prediction', 'accuracy_score', 'distance_to_closest_true_fire_km'], errors='ignore').to_dict()
        for key, value in new_properties.items():
            if isinstance(value, pd.Timestamp):
                new_properties[key] = value.isoformat()
        new_properties['accuracy_score'] = row['accuracy_score']
        new_properties['distance_to_closest_true_fire_km'] = row['distance_to_closest_true_fire_km']
        new_properties['display_prediction'] = row['display_prediction']

        new_feature = {
            "type": "Feature",
            "properties": new_properties,
            "geometry": row.geometry.__geo_interface__
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
    import os
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    true_fires_path = os.path.join(PROJECT_ROOT, "data/true_fires.geojson")
    metrics_file_path = os.path.join(PROJECT_ROOT, "static/metrics.json")

    # Calculate and update overall metrics for satellite_rule_based model
    predicted_fires_path_satellite = os.path.join(PROJECT_ROOT, "data/true_fires_with_weather_risk.geojson")
    calculate_and_update_metrics(true_fires_path, predicted_fires_path_satellite, metrics_file_path, model_name="satellite_rule_based")

    # Add accuracy to individual prediction files
    add_accuracy_to_predictions(true_fires_path, os.path.join(PROJECT_ROOT, "data/predicted_weather.geojson"), os.path.join(PROJECT_ROOT, "data/predicted_weather_with_accuracy.geojson"), start_date=None, end_date=None)
    add_accuracy_to_predictions(true_fires_path, os.path.join(PROJECT_ROOT, "data/predicted_baseline.geojson"), os.path.join(PROJECT_ROOT, "data/predicted_baseline_with_accuracy.geojson"), start_date=None, end_date=None)
    add_accuracy_to_predictions(true_fires_path, os.path.join(PROJECT_ROOT, "data/true_fires_with_weather_risk.geojson"), os.path.join(PROJECT_ROOT, "data/predicted_satellite_with_accuracy.geojson"), start_date=None, end_date=None)