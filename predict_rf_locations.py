
import pandas as pd
import joblib
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
# Use the same input data as the original prediction script for a fair comparison
ACTUAL_DATA_PATH = "data/fixed_actual.csv" 
MODEL_PATH = "models/random_forest_predictor.joblib"
PREDICTED_CSV_PATH = "data/rf_predict.csv"

def load_and_preprocess_data_for_prediction(data_path):
    """Loads and preprocesses the data for prediction (consistent with training scripts)."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    df = pd.read_csv(data_path, encoding='utf-8-sig')

    # --- Feature Engineering (Identical to training scripts) ---
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
        # NOTE: This re-fitting of LabelEncoder is not ideal but maintains consistency with the original script.
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df, features

if __name__ == "__main__":
    print("Loading data for RandomForest prediction...")
    actual_df, features = load_and_preprocess_data_for_prediction(ACTUAL_DATA_PATH)
    
    X_predict = actual_df[features]

    print("Loading trained RandomForest model...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained RandomForest model not found: {MODEL_PATH}. Please run train_random_forest.py first.")
    model = joblib.load(MODEL_PATH)

    print("Making predictions with RandomForest...")
    fire_probabilities = model.predict_proba(X_predict)[:, 1]

    # Filter for predicted fire locations (probability > 0.5)
    predicted_fire_indices = np.where(fire_probabilities > 0.5)[0]
    predicted_fire_locations_df = actual_df.iloc[predicted_fire_indices].copy()
    predicted_fire_locations_df['FIRE_PROBABILITY'] = fire_probabilities[predicted_fire_indices]

    # --- Save to CSV for map display ---
    os.makedirs(os.path.dirname(PREDICTED_CSV_PATH), exist_ok=True)
    if 'LAT' not in predicted_fire_locations_df.columns or 'LON' not in predicted_fire_locations_df.columns:
        print("WARNING: 'LAT' or 'LON' columns not found. Cannot generate rf_predict.csv for map.")
    else:
        predicted_fire_locations_df[['LAT', 'LON', 'FIRE_PROBABILITY']].to_csv(PREDICTED_CSV_PATH, index=False)
        print(f"RandomForest predicted fire locations saved to {PREDICTED_CSV_PATH}")

    print("RandomForest prediction process complete.")
