
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime, timedelta

# --- Configuration ---
ACTUAL_DATA_PATH = "data/fixed_actual.csv"
MODEL_PATH = "models/fire_prediction_model.joblib"
OUTPUT_JSON_PATH = "data/daily_top_future_predictions.json"
START_DATE = "2026-04-02"
END_DATE = "2027-04-02"

def generate_future_data(start_date_str, end_date_str, actual_data_path):
    """Generates a DataFrame for future predictions."""
    
    # Load actual data to get unique locations
    if not os.path.exists(actual_data_path):
        raise FileNotFoundError(f"Actual data file not found: {actual_data_path}")
    actual_df = pd.read_csv(actual_data_path, encoding='utf-8-sig')
    
    # Get unique locations (시도, 시군구) and their average lat/lon
    locations = actual_df[['발생장소_시도', '발생장소_시군구', 'LAT', 'LON']].drop_duplicates(subset=['발생장소_시도', '발생장소_시군구'])
    
    # Generate date range
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    
    future_data = []
    for date in date_range:
        # To reduce computation, we might only predict for peak hours or a subset of locations
        # But for now, let's follow the previous logic with fixed hours
        for hour in [10, 14, 18]: # Predicting for representative hours
            for _, loc_row in locations.iterrows():
                future_data.append({
                    'latitude': loc_row['LAT'],
                    'longitude': loc_row['LON'],
                    'month': date.month,
                    'day': date.day,
                    'dayofweek': date.weekday(),
                    'hour': hour,
                    'date_str': date.strftime('%Y%m%d'),
                    'location_sido': loc_row['발생장소_시도'],
                    'location_sigungu': loc_row['발생장소_시군구']
                })
    
    return pd.DataFrame(future_data)

if __name__ == "__main__":
    print("Generating future data for prediction...")
    future_df = generate_future_data(START_DATE, END_DATE, ACTUAL_DATA_PATH)
    
    features_for_prediction = ['latitude', 'longitude', 'month', 'day', 'dayofweek', 'hour']
    
    X_future = future_df[features_for_prediction]
    
    print("Loading trained model...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}. Please train the model first.")
    model = joblib.load(MODEL_PATH)
    
    print("Making future predictions...")
    future_probabilities = model.predict_proba(X_future)[:, 1]
    
    future_df['FIRE_PROBABILITY'] = future_probabilities
    
    # Group by date and get the top prediction (highest probability) for each day
    daily_top_predictions = future_df.loc[future_df.groupby('date_str')['FIRE_PROBABILITY'].idxmax()]

    # --- Save to JSON ---
    output_predictions = []
    for _, row in daily_top_predictions.iterrows():
        output_predictions.append({
            "date": row['date_str'],
            "lat": row['latitude'],
            "lon": row['longitude'],
            "probability": float(row['FIRE_PROBABILITY']),
            "location_sido": row['location_sido'],
            "location_sigungu": row['location_sigungu']
        })
        
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_predictions, f, ensure_ascii=False, indent=4)
        
    print(f"Daily top future predictions saved to {OUTPUT_JSON_PATH}")
    print("Prediction generation complete.")
