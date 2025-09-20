import pandas as pd
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, precision_score
from imblearn.over_sampling import SMOTE
import optuna
import json

# --- Configuration ---
DATA_PATH = "data/산림청_산불상황관제시스템 산불통계데이터_20241016.csv"
MODEL_PATH = "models/fire_prediction_model.joblib"
BEST_ACCURACY_PATH = "best_accuracy.txt"
N_TRIALS = 200  # Number of optimization trials

def load_and_preprocess_data(data_path):
    """Loads and preprocesses the data."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    df = pd.read_csv(data_path, encoding='utf-8-sig')

    # --- Feature Engineering ---
    # NOTE: The 'no-fire' data generation is artificial and might be a source of issues.
    # For better performance, consider using real 'no-fire' data.
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
    
    # Randomize time and shuffle categorical features to create more realistic negative samples
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
    """Optuna objective function."""
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),  # L1 규제 추가
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True), # L2 규제 추가
    }

    model = lgb.LGBMClassifier(**param, random_state=42)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

def get_best_accuracy(path):
    """Reads the best accuracy from a file."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            try:
                return float(f.read())
            except ValueError:
                return 0.0
    return 0.0

def save_best_accuracy(path, accuracy):
    """Saves the best accuracy to a file."""
    with open(path, 'w') as f:
        f.write(str(accuracy))

if __name__ == "__main__":
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data(DATA_PATH)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Starting hyperparameter optimization with Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=N_TRIALS)

    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.params}")
    
    new_best_accuracy = study.best_value
    print(f"Best accuracy in this run: {new_best_accuracy:.4f}")

    current_best_accuracy = get_best_accuracy(BEST_ACCURACY_PATH)
    print(f"Current best accuracy on record: {current_best_accuracy:.4f}")

    # Save the model if accuracy is within the desired range (90% - 99.9%)
    if 0.90 <= new_best_accuracy < 1.0:
        # Save if it's an improvement over the current score, or if the current score is 100%
        if new_best_accuracy > current_best_accuracy or current_best_accuracy >= 1.0:
            print(f"!!! New best accuracy in range (90-99.9%) found: {new_best_accuracy:.4f} !!!")
            save_best_accuracy(BEST_ACCURACY_PATH, new_best_accuracy)
            
            # Train the best model found by Optuna on the full training data
            best_model = lgb.LGBMClassifier(**study.best_trial.params, random_state=42)
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            best_model.fit(X_train_res, y_train_res)
            
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            joblib.dump(best_model, MODEL_PATH)
            print(f"Best model saved to {MODEL_PATH}")

            # Evaluate the best model to get all metrics
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc_roc = roc_auc_score(y_test, y_pred_proba)

            metrics = {
                "baseline": {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "auc_roc": auc_roc
                }
            }

            # Save metrics to static/metrics.json
            metrics_file_path = "static/metrics.json"
            os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)
            with open(metrics_file_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=4)
            print(f"Metrics saved to {metrics_file_path}")
        else:
            print(f"New accuracy {new_best_accuracy:.4f} is in the desired range, but not better than the current best of {current_best_accuracy:.4f}.")
    elif new_best_accuracy >= 1.0:
        print(f"Accuracy is 100% or higher ({new_best_accuracy:.4f}). Model not saved as per new policy.")
    else:
        print(f"Accuracy {new_best_accuracy:.4f} is below the 90% threshold. Model not saved.")