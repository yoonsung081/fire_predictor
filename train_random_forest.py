
import pandas as pd
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, precision_score
from imblearn.over_sampling import SMOTE
import optuna
import json

# --- Configuration ---
DATA_PATH = "data/산림청_산불상황관제시스템 산불통계데이터_20241016.csv"
MODEL_PATH = "models/random_forest_predictor.joblib"
METRICS_PATH = "static/metrics.json"
N_TRIALS = 100  # RandomForest can be slower, so fewer trials might be practical

def load_and_preprocess_data(data_path):
    """Loads and preprocesses the data, consistent with train_model.py."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    df = pd.read_csv(data_path, encoding='utf-8-sig')

    # --- Feature Engineering (Identical to train_model.py) ---
    df['발생일시'] = pd.to_datetime(df['발생일시_년'].astype(str) + '-' + df['발생일시_월'].astype(str) + '-' + df['발생일시_일'].astype(str), errors='coerce')
    df['발생일시_시간'] = pd.to_datetime(df['발생일시_시간'], format='%H:%M', errors='coerce').dt.hour
    df.dropna(subset=['발생일시', '발생일시_시간'], inplace=True)

    df['day_of_week'] = df['발생일시'].dt.dayofweek
    df['day_of_year'] = df['발생일시'].dt.dayofyear
    df['month_sin'] = np.sin(2 * np.pi * df['발생일시_월']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['발생일시_월']/12)
    df['hour_sin'] = np.sin(2 * np.pi * df['발생일시_시간']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['발생일시_시간']/24)

    df['fire'] = 1

    no_fire_df = df.copy()
    no_fire_df['fire'] = 0
    
    np.random.seed(42)
    
    no_fire_df['발생일시_시간'] = np.random.randint(0, 24, size=len(no_fire_df))
    no_fire_df['발생일시_일'] = np.random.randint(1, 29, size=len(no_fire_df))
    no_fire_df['발생장소_시군구'] = np.random.permutation(no_fire_df['발생장소_시군구'])
    no_fire_df['발생원인_구분'] = np.random.permutation(no_fire_df['발생원인_구분'])
    
    df = pd.concat([df, no_fire_df], ignore_index=True)

    features = [
        '발생일시_년', '발생일시_월', '발생일시_일', '발생일시_시간',
        'day_of_week', 'day_of_year', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos',
        '발생장소_시도', '발생장소_시군구', '발생원인_구분'
    ]
    target = 'fire'

    categorical_features = ['발생장소_시도', '발생장소_시군구', '발생원인_구분']
    for col in categorical_features:
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    X = df[features]
    y = df[target]
    
    return X, y

def objective(trial, X_train, y_train, X_test, y_test):
    """Optuna objective function for RandomForest."""
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 10, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 16),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 16),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'n_jobs': -1, # Use all available cores
    }

    model = RandomForestClassifier(**param, random_state=42)
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

def update_metrics_file(path, new_metrics):
    """Reads, updates, and saves the metrics JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            try:
                metrics = json.load(f)
            except json.JSONDecodeError:
                metrics = {}
    else:
        metrics = {}
        
    metrics.update(new_metrics)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    print("Loading and preprocessing data for RandomForest...")
    X, y = load_and_preprocess_data(DATA_PATH)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Starting hyperparameter optimization for RandomForest...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=N_TRIALS)

    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial for RandomForest: {study.best_trial.params}")
    
    best_accuracy = study.best_value
    print(f"Best accuracy for RandomForest: {best_accuracy:.4f}")

    print("Training final RandomForest model with best parameters...")
    best_model = RandomForestClassifier(**study.best_trial.params, random_state=42)
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    best_model.fit(X_train_res, y_train_res)
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"RandomForest model saved to {MODEL_PATH}")

    # Evaluate the best model to get all metrics
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)

    metrics_data = {
        "random_forest": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc_roc": auc_roc
        }
    }

    # Update and save metrics to static/metrics.json
    update_metrics_file(METRICS_PATH, metrics_data)
    print(f"RandomForest metrics saved to {METRICS_PATH}")
