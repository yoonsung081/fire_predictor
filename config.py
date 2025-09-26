"""
Configuration file for the fire predictor project.
"""

import os

# --- Project Root ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Data Paths ---
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "산림청_산불상황관제시스템 산불통계데이터_20241016.csv")
PREPROCESSED_DATA_PATH = os.path.join(DATA_DIR, "with_coordinates.csv")
ACTUAL_DATA_PATH = os.path.join(DATA_DIR, "fixed_actual.csv")
PREDICTED_CSV_PATH = os.path.join(DATA_DIR, "fixed_predict.csv")
PREDICTED_JSON_PATH = os.path.join(DATA_DIR, "predicted_fire_markers.json")

# --- Model Paths ---
MODELS_DIR = os.path.join(ROOT_DIR, "models")
LGBM_MODEL_PATH = os.path.join(MODELS_DIR, "fire_prediction_model.joblib")
TRANSFORMER_MODEL_PATH = os.path.join(MODELS_DIR, "transformer_predictor.pth")
SCALER_PATH = os.path.join(MODELS_DIR, "transformer_scaler.joblib")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoders.joblib")


# --- Metrics & Performance ---
STATIC_DIR = os.path.join(ROOT_DIR, "static")
METRICS_PATH = os.path.join(STATIC_DIR, "metrics.json")
BEST_ACCURACY_PATH = os.path.join(ROOT_DIR, "best_accuracy.txt")

# --- Model Training Parameters ---
N_TRIALS_OPTUNA = 200
TRANSFORMER_SEQ_LENGTH = 30
TRANSFORMER_PRED_LENGTH = 7
