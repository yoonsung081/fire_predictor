"""
Prediction functions for the fire predictor project.
"""

import os
import pandas as pd
import numpy as np
import joblib
import json

import config
from src.data_processing import preprocess_data

def predict_lgbm(data_path):
    """
    Runs the LightGBM prediction process.

    Args:
        data_path (str): Path to the data for prediction.

    Returns:
        pd.DataFrame: DataFrame with prediction results.
    """
    print("Loading and preprocessing data for prediction...")
    X_predict, df = preprocess_data(data_path, mode='predict')

    print("Loading trained LightGBM model...")
    if not os.path.exists(config.LGBM_MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found: {config.LGBM_MODEL_PATH}")
    model = joblib.load(config.LGBM_MODEL_PATH)

    print("Making predictions...")
    fire_probabilities = model.predict_proba(X_predict)[:, 1]

    df['fire_probability'] = fire_probabilities
    df_pred = df[fire_probabilities > 0.5].copy()

    # --- Generate JSON output ---
    predicted_markers = []
    for _, row in df_pred.iterrows():
        if 'LAT' in row and 'LON' in row:
            predicted_markers.append({
                "lat": row['LAT'],
                "lon": row['LON'],
                "probability": row['fire_probability']
            })
    
    os.makedirs(os.path.dirname(config.PREDICTED_JSON_PATH), exist_ok=True)
    with open(config.PREDICTED_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(predicted_markers, f, ensure_ascii=False, indent=4)
    print(f"Predicted fire markers saved to {config.PREDICTED_JSON_PATH}")

    return df_pred
