
import pandas as pd
import numpy as np
import json
import os
from sklearn.linear_model import LinearRegression

def analyze_trend(data_path, output_path):
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    # Load data
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    
    # Create date column
    df['date'] = pd.to_datetime(df['발생일시_년'].astype(str) + '-' + 
                                df['발생일시_월'].astype(str) + '-' + 
                                df['발생일시_일'].astype(str))
    
    # Group by date to count fires
    daily_counts = df.groupby('date').size().reset_index(name='count')
    daily_counts = daily_counts.sort_values('date')
    
    # Prepare data for linear regression
    # Use ordinal dates for regression
    X = np.array(range(len(daily_counts))).reshape(-1, 1)
    y = daily_counts['count'].values
    
    model = LinearRegression()
    model.fit(X, y)
    trend_line = model.predict(X)
    
    # Prepare output data
    result = {
        "dates": daily_counts['date'].dt.strftime('%Y-%m-%d').tolist(),
        "counts": daily_counts['count'].tolist(),
        "trend": trend_line.tolist(),
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_)
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    print(f"Trend analysis saved to {output_path}")

if __name__ == "__main__":
    data_path = "data/fixed_actual.csv"
    output_path = "static/fire_trend.json"
    analyze_trend(data_path, output_path)
