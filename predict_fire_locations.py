import pandas as pd
import joblib
import numpy as np
import os
import json
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
ACTUAL_DATA_PATH = "data/fixed_actual.csv"
MODEL_PATH = "models/fire_prediction_model.joblib"
PREDICTED_CSV_PATH = "data/fixed_predict.csv"
PREDICTED_JSON_PATH = "data/predicted_fire_markers.json"

def load_and_preprocess_data_for_prediction(data_path):
    """Loads and preprocesses the data for prediction."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    df = pd.read_csv(data_path, encoding='utf-8-sig')

    # --- Feature Engineering (consistent with training) ---
    df['발생일시'] = pd.to_datetime(df['발생일시_년'].astype(str) + '-' + df['발생일시_월'].astype(str) + '-' + df['발생일시_일'].astype(str), errors='coerce')
    df['발생일시_시간'] = pd.to_datetime(df['발생일시_시간'], format='%H:%M', errors='coerce').dt.hour
    df.dropna(subset=['발생일시', '발생일시_시간'], inplace=True)

    df['day_of_week'] = df['발생일시'].dt.dayofweek
    df['day_of_year'] = df['발생일시'].dt.dayofyear
    df['month_sin'] = np.sin(2 * np.pi * df['발생일시_월']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['발생일시_월']/12)
    df['hour_sin'] = np.sin(2 * np.pi * df['발생일시_시간']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['발생일시_시간']/24)

    features = [
        '발생일시_년', '발생일시_월', '발생일시_일', '발생일시_시간',
        'day_of_week', 'day_of_year', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos',
        '발생장소_시도', '발생장소_시군구', '발생원인_구분'
    ]

    categorical_features = ['발생장소_시도', '발생장소_시군구', '발생원인_구분']
    for col in categorical_features:
        df[col] = df[col].astype(str)
        # WARNING: LabelEncoder should ideally be fitted on training data and then transformed here.
        # Refitting on prediction data might lead to inconsistencies if categories differ.
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df, features

if __name__ == "__main__":
    print("Loading and preprocessing actual wildfire data...")
    actual_df, features = load_and_preprocess_data_for_prediction(ACTUAL_DATA_PATH)
    
    X_predict = actual_df[features]

    print("Loading trained model...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}. Please train the model first.")
    model = joblib.load(MODEL_PATH)

    print("Making predictions...")
    # Get probability of fire (class 1)
    fire_probabilities = model.predict_proba(X_predict)[:, 1]

    # Filter for predicted fire locations (e.g., probability > 0.5)
    # You might want to adjust this threshold based on desired sensitivity/specificity
    predicted_fire_indices = np.where(fire_probabilities > 0.5)[0]
    predicted_fire_locations_df = actual_df.iloc[predicted_fire_indices].copy()
    predicted_fire_locations_df['FIRE_PROBABILITY'] = fire_probabilities[predicted_fire_indices]

    # --- Save to CSV for generate_map.py ---
    os.makedirs(os.path.dirname(PREDICTED_CSV_PATH), exist_ok=True)
    # Ensure LAT and LON columns exist for generate_map.py
    if 'LAT' not in predicted_fire_locations_df.columns or 'LON' not in predicted_fire_locations_df.columns:
        print("WARNING: 'LAT' or 'LON' columns not found in actual data. Cannot generate fixed_predict.csv for map.")
    else:
        predicted_fire_locations_df[['LAT', 'LON', 'FIRE_PROBABILITY']].to_csv(PREDICTED_CSV_PATH, index=False)
        print(f"Predicted fire locations saved to {PREDICTED_CSV_PATH}")

    # --- Generate JSON output for user ---
    predicted_markers = []
    for index, row in predicted_fire_locations_df.iterrows():
        if 'LAT' in row and 'LON' in row:
            predicted_markers.append({
                "lat": row['LAT'],
                "lon": row['LON'],
                "probability": row['FIRE_PROBABILITY']
            })
    
    os.makedirs(os.path.dirname(PREDICTED_JSON_PATH), exist_ok=True)
    with open(PREDICTED_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(predicted_markers, f, ensure_ascii=False, indent=4)
    print(f"Predicted fire markers saved to {PREDICTED_JSON_PATH}")

    print("Prediction process complete.")
