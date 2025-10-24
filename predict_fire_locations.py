import pandas as pd
import joblib
import numpy as np
import os
import json

from src.data_processing import preprocess_data

ACTUAL_DATA_PATH = "data/fixed_actual.csv"
LGBM_MODEL_PATH = "models/lgbm_predictor.joblib"
PREDICTED_CSV_PATH = "data/lgbm_predict.csv"
PREDICTED_JSON_PATH = "static/lgbm_predictions.json"


if __name__ == "__main__":
    print("Loading and preprocessing actual wildfire data for prediction...")
    X_predict, actual_df = preprocess_data(ACTUAL_DATA_PATH, mode='predict')

    print("Loading trained model...")
    if not os.path.exists(LGBM_MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found: {LGBM_MODEL_PATH}. Please train the model first.")
    model = joblib.load(LGBM_MODEL_PATH)

    print("Making predictions...")
    fire_probabilities = model.predict_proba(X_predict)[:, 1]

    predicted_fire_indices = np.where(fire_probabilities > 0.5)[0]
    predicted_fire_locations_df = actual_df.iloc[predicted_fire_indices].copy()
    predicted_fire_locations_df['FIRE_PROBABILITY'] = fire_probabilities[predicted_fire_indices]

    os.makedirs(os.path.dirname(PREDICTED_CSV_PATH), exist_ok=True)
    if 'LAT' not in predicted_fire_locations_df.columns or 'LON' not in predicted_fire_locations_df.columns:
        print("WARNING: 'LAT' or 'LON' columns not found in actual data. Cannot generate fixed_predict.csv for map.")
    else:
        predicted_fire_locations_df[['LAT', 'LON', 'FIRE_PROBABILITY']].to_csv(PREDICTED_CSV_PATH, index=False)
        print(f"Predicted fire locations saved to {PREDICTED_CSV_PATH}")

    predicted_markers = []
    for index, row in predicted_fire_locations_df.iterrows():
        if 'LAT' in row and 'LON' in row:
            predicted_markers.append({
                "lat": row['LAT'],
                "lon": row['LON'],
                "fire_probability": row['FIRE_PROBABILITY']
            })
    
    os.makedirs(os.path.dirname(PREDICTED_JSON_PATH), exist_ok=True)
    with open(PREDICTED_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(predicted_markers, f, ensure_ascii=False, indent=4)
    print(f"Predicted fire markers saved to {PREDICTED_JSON_PATH}")

    print("Prediction process complete.")