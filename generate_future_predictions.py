
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import LabelEncoder
import os
from datetime import datetime, timedelta

# --- Configuration ---
ACTUAL_DATA_PATH = "data/fixed_actual.csv"
MODEL_PATH = "models/fire_prediction_model.joblib"
OUTPUT_JSON_PATH = "data/daily_top_future_predictions.json"
START_DATE = "2024-09-29"
END_DATE = "2025-12-31"

def generate_future_data(start_date_str, end_date_str, actual_data_path):
    """Generates a DataFrame for future predictions."""
    
    # Load actual data to get unique locations
    if not os.path.exists(actual_data_path):
        raise FileNotFoundError(f"Actual data file not found: {actual_data_path}")
    actual_df = pd.read_csv(actual_data_path, encoding='utf-8-sig')
    
    # Get unique locations (시도, 시군구) and their average lat/lon
    locations = actual_df[['발생장소_시도', '발생장소_시군구', 'LAT', 'LON']].drop_duplicates(subset=['발생장소_시도', '발생장소_시군구'])
    
    # Get unique causes
    causes = actual_df['발생원인_구분'].unique()

    # Generate date range
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    
    future_data = []
    for date in date_range:
        for hour in range(24):
            for _, loc_row in locations.iterrows():
                for cause in causes:
                    future_data.append({
                        '발생일시_년': date.year,
                        '발생일시_월': date.month,
                        '발생일시_일': date.day,
                        '발생일시_시간': hour,
                        '발생장소_시도': loc_row['발생장소_시도'],
                        '발생장소_시군구': loc_row['발생장소_시군구'],
                        '발생원인_구분': cause,
                        'LAT': loc_row['LAT'],
                        'LON': loc_row['LON']
                    })
    
    return pd.DataFrame(future_data)

if __name__ == "__main__":
    print("Generating future data for prediction...")
    future_df = generate_future_data(START_DATE, END_DATE, ACTUAL_DATA_PATH)
    
    print("Preprocessing future data...")
    # --- Feature Engineering (consistent with training) ---
    future_df['발생일시'] = pd.to_datetime(future_df['발생일시_년'].astype(str) + '-' + future_df['발생일시_월'].astype(str) + '-' + future_df['발생일시_일'].astype(str))
    future_df['day_of_week'] = future_df['발생일시'].dt.dayofweek
    future_df['day_of_year'] = future_df['발생일시'].dt.dayofyear
    future_df['month_sin'] = np.sin(2 * np.pi * future_df['발생일시_월']/12)
    future_df['month_cos'] = np.cos(2 * np.pi * future_df['발생일시_월']/12)
    future_df['hour_sin'] = np.sin(2 * np.pi * future_df['발생일시_시간']/24)
    future_df['hour_cos'] = np.cos(2 * np.pi * future_df['발생일시_시간']/24)

    features_to_encode = ['발생장소_시도', '발생장소_시군구', '발생원인_구분']
    
    # Load the actual data again to fit the encoders to ensure consistency
    actual_df_for_encoding = pd.read_csv(ACTUAL_DATA_PATH, encoding='utf-8-sig')
    
    for col in features_to_encode:
        le = LabelEncoder()
        # Fit on the actual data
        le.fit(actual_df_for_encoding[col].astype(str))
        # Transform the future data
        future_df[col] = le.transform(future_df[col].astype(str))

    features_for_prediction = [
        '발생일시_년', '발생일시_월', '발생일시_일', '발생일시_시간',
        'day_of_week', 'day_of_year', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos',
        '발생장소_시도', '발생장소_시군구', '발생원인_구분'
    ]
    
    X_future = future_df[features_for_prediction]
    
    print("Loading trained model...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}. Please train the model first.")
    model = joblib.load(MODEL_PATH)
    
    print("Making future predictions...")
    future_probabilities = model.predict_proba(X_future)[:, 1]
    
    future_df['FIRE_PROBABILITY'] = future_probabilities
    
    # Filter for predictions with a probability > 0.5 (or another threshold)
    high_prob_predictions = future_df[future_df['FIRE_PROBABILITY'] > 0.5].copy()
    
    # Group by date and get the top prediction (highest probability) for each day
    daily_top_predictions = high_prob_predictions.loc[high_prob_predictions.groupby(high_prob_predictions['발생일시'].dt.date)['FIRE_PROBABILITY'].idxmax()]

    # --- Save to JSON ---
    output_predictions = []
    for _, row in daily_top_predictions.iterrows():
        output_predictions.append({
            "date": row['발생일시'].strftime('%Y%m%d'),
            "lat": row['LAT'],
            "lon": row['LON'],
            "probability": row['FIRE_PROBABILITY'],
            "location_sido": row['발생장소_시도'], # Add location info
            "location_sigungu": row['발생장소_시군구'] # Add location info
        })
        
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_predictions, f, ensure_ascii=False, indent=4)
        
    print(f"Daily top future predictions saved to {OUTPUT_JSON_PATH}")
    print("Prediction generation complete.")
