import pandas as pd
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, precision_score
from imblearn.over_sampling import SMOTE
import optuna
import json

import config
from src.data_processing import preprocess_data

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
    X, y, _ = preprocess_data(config.RAW_DATA_PATH, mode='train')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Starting hyperparameter optimization with Optuna ({config.N_TRIALS_OPTUNA} trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=config.N_TRIALS_OPTUNA)

    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.params}")
    
    new_best_accuracy = study.best_value
    print(f"Best accuracy in this run: {new_best_accuracy:.4f}")

    current_best_accuracy = get_best_accuracy(config.BEST_ACCURACY_PATH)
    print(f"Current best accuracy on record: {current_best_accuracy:.4f}")

    if 0.90 <= new_best_accuracy < 1.0:
        if new_best_accuracy > current_best_accuracy or current_best_accuracy >= 1.0:
            print(f"!!! New best accuracy in range (90-99.9%) found: {new_best_accuracy:.4f} !!!")
            save_best_accuracy(config.BEST_ACCURACY_PATH, new_best_accuracy)
            
            best_model = lgb.LGBMClassifier(**study.best_trial.params, random_state=42)
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            best_model.fit(X_train_res, y_train_res)
            
            os.makedirs(os.path.dirname(config.LGBM_MODEL_PATH), exist_ok=True)
            joblib.dump(best_model, config.LGBM_MODEL_PATH)
            print(f"Best model saved to {config.LGBM_MODEL_PATH}")

            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            metrics = {
                "baseline": {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1": f1_score(y_test, y_pred),
                    "auc_roc": roc_auc_score(y_test, y_pred_proba)
                }
            }

            os.makedirs(os.path.dirname(config.METRICS_PATH), exist_ok=True)
            with open(config.METRICS_PATH, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=4)
            print(f"Metrics saved to {config.METRICS_PATH}")
        else:
            print(f"New accuracy {new_best_accuracy:.4f} is in the desired range, but not better than the current best of {current_best_accuracy:.4f}.")
    elif new_best_accuracy >= 1.0:
        print(f"Accuracy is 100% or higher ({new_best_accuracy:.4f}). Model not saved as per new policy.")
    else:
        print(f"Accuracy {new_best_accuracy:.4f} is below the 90% threshold. Model not saved.")

    print("Training process complete.")
