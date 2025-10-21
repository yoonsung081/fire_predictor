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

def predict_lgbm(data_path, output_filename):
    """
    Runs the LightGBM prediction process.

    Args:
        data_path (str): Path to the data for prediction.
        output_filename (str): The name of the output JSON file.

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
    export_data_to_json(df_pred, output_filename, extra_cols=['fire_probability'])

    return df_pred
