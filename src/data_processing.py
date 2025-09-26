"""
Data processing functions for the fire predictor project.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

import config

def preprocess_data(data_path, mode='train'):
    """
    Loads and preprocesses data for training or prediction.

    Args:
        data_path (str): Path to the raw data file.
        mode (str): 'train' or 'predict'. 
                    In 'train' mode, it fits and saves label encoders.
                    In 'predict'mode, it loads and uses saved encoders.

    Returns:
        pd.DataFrame: Processed DataFrame with features.
        pd.Series: Target variable (only in 'train' mode).
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path, encoding='utf-8-sig')

    # --- Consistent Feature Engineering ---
    df['발생일시'] = pd.to_datetime(
        df['발생일시_년'].astype(str) + '-' + 
        df['발생일시_월'].astype(str) + '-' + 
        df['발생일시_일'].astype(str), 
        errors='coerce'
    )
    df['발생일시_시간'] = pd.to_datetime(df['발생일시_시간'], format='%H:%M', errors='coerce').dt.hour
    df.dropna(subset=['발생일시', '발생일시_시간'], inplace=True)

    df['day_of_week'] = df['발생일시'].dt.dayofweek
    df['day_of_year'] = df['발생일시'].dt.dayofyear
    df['month_sin'] = np.sin(2 * np.pi * df['발생일시_월'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['발생일시_월'] / 12)
    df['hour_sin'] = np.sin(2 * np.pi * df['발생일시_시간'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['발생일시_시간'] / 24)

    categorical_features = ['발생장소_시도', '발생장소_시군구', '발생원인_구분']
    encoders = {}

    if mode == 'train':
        # Create and fit encoders, then save them
        for col in categorical_features:
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        joblib.dump(encoders, config.LABEL_ENCODER_PATH)
        print(f"Label encoders saved to {config.LABEL_ENCODER_PATH}")

    elif mode == 'predict':
        # Load saved encoders
        if not os.path.exists(config.LABEL_ENCODER_PATH):
            raise FileNotFoundError(f"Label encoder file not found at {config.LABEL_ENCODER_PATH}. Please train the model first.")
        encoders = joblib.load(config.LABEL_ENCODER_PATH)
        for col in categorical_features:
            df[col] = df[col].astype(str)
            le = encoders.get(col)
            if le:
                # Handle unseen labels in prediction data
                df[col] = df[col].apply(lambda s: s if s in le.classes_ else 'unseen')
                le.classes_ = np.append(le.classes_, 'unseen')
                df[col] = le.transform(df[col])
            else:
                raise ValueError(f"Encoder for column '{col}' not found.")
    else:
        raise ValueError("Mode must be either 'train' or 'predict'.")

    # --- Target and Features --- 
    features = [
        '발생일시_년', '발생일시_월', '발생일시_일', '발생일시_시간',
        'day_of_week', 'day_of_year', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos',
        '발생장소_시도', '발생장소_시군구', '발생원인_구분'
    ]

    if mode == 'train':
        # Artificial negative sampling for training
        df['fire'] = 1
        no_fire_df = df.copy()
        no_fire_df['fire'] = 0
        np.random.seed(42)
        no_fire_df['발생일시_시간'] = np.random.randint(0, 24, size=len(no_fire_df))
        no_fire_df['발생일시_일'] = np.random.randint(1, 29, size=len(no_fire_df))
        no_fire_df['발생장소_시군구'] = np.random.permutation(no_fire_df['발생장소_시군구'])
        no_fire_df['발생원인_구분'] = np.random.permutation(no_fire_df['발생원인_구분'])
        df = pd.concat([df, no_fire_df], ignore_index=True)
        
        X = df[features]
        y = df['fire']
        return X, y, df # Return original df for other uses
    else:
        X = df[features]
        return X, df # Return original df for other uses
