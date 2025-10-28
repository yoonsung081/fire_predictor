
import json
import pandas as pd
from datetime import datetime

def prepare_fire_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        fire_data = json.load(f)

    df = pd.DataFrame(fire_data)

    # Convert start_time to datetime objects
    df['start_time'] = pd.to_datetime(df['start_time'], format='%Y-%m-%d %H:%M')

    # Extract features
    df['month'] = df['start_time'].dt.month
    df['day'] = df['start_time'].dt.day
    df['dayofweek'] = df['start_time'].dt.dayofweek
    df['hour'] = df['start_time'].dt.hour

    # Target variable: is_fire (all entries in this dataset are fires, so 1)
    df['is_fire'] = 1

    # Select relevant features for the model
    features = df[['latitude', 'longitude', 'month', 'day', 'dayofweek', 'hour', 'is_fire', 'start_time']]
    return features

if __name__ == "__main__":
    actual_data_path = r"C:\Users\000\OneDrive\Desktop\fire_predictor_project\fire_data_with_updated_coords.json"
    prepared_data = prepare_fire_data(actual_data_path)
    print(prepared_data.head())
    
    # Save prepared data to a CSV for easier use in the next step
    prepared_data.to_csv(r"C:\Users\000\OneDrive\Desktop\fire_predictor_project\data\prepared_fire_data.csv", index=False, encoding='utf-8')
    print("Prepared data saved to data/prepared_fire_data.csv")
