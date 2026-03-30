
import pandas as pd
import numpy as np
import json
import os

def generate_heatmap_data(data_path, output_path):
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path, encoding='utf-8-sig')
    
    # 시간 정보 처리
    if df['발생일시_시간'].dtype == object:
        df['hour'] = pd.to_datetime(df['발생일시_시간'], format='%H:%M', errors='coerce').dt.hour
    else:
        df['hour'] = df['발생일시_시간']
    
    df = df.dropna(subset=['발생일시_월', 'hour'])
    
    # 월(1-12) x 시간(0-23) 매트릭스 생성
    heatmap_matrix = np.zeros((12, 24), dtype=int)
    
    for _, row in df.iterrows():
        m = int(row['발생일시_월']) - 1 # 0-indexed
        h = int(row['hour'])
        if 0 <= m < 12 and 0 <= h < 24:
            heatmap_matrix[m, h] += 1
            
    # JSON 저장을 위해 리스트로 변환
    result = {
        "matrix": heatmap_matrix.tolist(),
        "max_val": int(heatmap_matrix.max()),
        "months": [f"{i}월" for i in range(1, 13)],
        "hours": [f"{i}시" for i in range(24)]
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    print(f"Heatmap data saved to {output_path}")

if __name__ == "__main__":
    data_path = "data/fixed_actual.csv"
    output_path = "static/heatmap_data.json"
    generate_heatmap_data(data_path, output_path)
