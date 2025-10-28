
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import json

def generate_and_update_predictions(model_path, output_json_path, actual_data_for_range_path):
    # Load the trained model
    model = joblib.load(model_path)

    # Load actual data to determine the date and spatial range for predictions
    with open(actual_data_for_range_path, 'r', encoding='utf-8') as f:
        actual_fires = json.load(f)
    actual_df = pd.DataFrame(actual_fires)
    actual_df['start_time'] = pd.to_datetime(actual_df['start_time'])

    # Determine the date range for predictions (all unique dates from actual fire data)
    prediction_dates = actual_df['start_time'].dt.date.unique()
    prediction_dates = sorted(list(prediction_dates)) # Sort dates for consistent output

    # Determine the spatial range for predictions (based on actual fire data)
    min_lat, max_lat = actual_df['latitude'].min(), actual_df['latitude'].max()
    min_lon, max_lon = actual_df['longitude'].min(), actual_df['longitude'].max()

    # Create a grid of latitude and longitude for predictions
    # Adjust step size for a reasonable number of prediction points
    lat_step = (max_lat - min_lat) / 20 # 20 steps in latitude
    lon_step = (max_lon - min_lon) / 20 # 20 steps in longitude

    lats = np.arange(min_lat, max_lat, lat_step)
    lons = np.arange(min_lon, max_lon, lon_step)

    # Generate future prediction data points
    future_predictions_list = []

    for pred_date in prediction_dates:
        for lat in lats:
            for lon in lons:
                # Create features for prediction
                features = pd.DataFrame([{
                    'latitude': lat,
                    'longitude': lon,
                    'month': pred_date.month,
                    'day': pred_date.day,
                    'dayofweek': pred_date.weekday(),
                    'hour': 12 # Assuming a default hour for prediction, e.g., noon
                }])
                
                # Predict probability of fire
                probability = model.predict_proba(features)[0][1]
                
                future_predictions_list.append({
                    "date": pred_date.strftime("%Y%m%d"),
                    "lat": lat,
                    "lon": lon,
                    "probability": probability
                })

    # Group by date and select the top prediction (highest probability) for each day
    predictions_df = pd.DataFrame(future_predictions_list)
    probability_threshold = 0.5 # You can adjust this threshold
    filtered_predictions = predictions_df[predictions_df['probability'] >= probability_threshold]

    # Convert to the desired JSON format
    output_data = filtered_predictions[['date', 'lat', 'lon', 'probability']].to_dict(orient='records')

    # Update the daily_top_future_predictions.json file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"Updated predictions saved to {output_json_path}")

if __name__ == "__main__":
    model_path = r"C:\Users\000\OneDrive\Desktop\fire_predictor_project\models\fire_prediction_model.joblib"
    output_json_path = r"C:\Users\000\OneDrive\Desktop\fire_predictor_project\data\daily_top_future_predictions.json"
    actual_data_for_range_path = r"C:\Users\000\OneDrive\Desktop\fire_predictor_project\fire_data_with_updated_coords.json"
    
    generate_and_update_predictions(model_path, output_json_path, actual_data_for_range_path)
