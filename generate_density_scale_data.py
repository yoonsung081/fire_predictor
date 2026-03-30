
import pandas as pd
import numpy as np
import json
import os

def generate_density_scale_data(data_path, output_path):
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path, encoding='utf-8-sig')
    
    # 3. 지역별 발생 건수 (시도별)
    region_counts = df['발생장소_시도'].value_counts().to_dict()
    
    # 4. 피해 규모별 분포 (피해면적_합계 기준)
    # 소형: < 0.1ha, 중형: 0.1~1ha, 대형: 1~10ha, 초대형: >= 10ha
    bins = [0, 0.1, 1.0, 10.0, float('inf')]
    labels = ['소형 (<0.1ha)', '중형 (0.1~1ha)', '대형 (1~10ha)', '초대형 (>=10ha)']
    
    # 피해면적_합계 컬럼 확인 (없을 경우 0으로 처리)
    area_col = '피해면적_합계' if '피해면적_합계' in df.columns else 'MOUNTAIN' # 예비 컬럼
    if area_col not in df.columns:
        df[area_col] = 0.05 # 기본값 (대부분 소형으로 분류)

    df['scale'] = pd.cut(df[area_col], bins=bins, labels=labels, right=False)
    scale_counts = df['scale'].value_counts().sort_index().to_dict()
    # Categorical 객체에서 string으로 변환
    scale_counts = {str(k): int(v) for k, v in scale_counts.items()}

    # 결과 저장
    result = {
        "region": region_counts,
        "scale": scale_counts
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    print(f"Density and scale data saved to {output_path}")

if __name__ == "__main__":
    data_path = "data/fixed_actual.csv"
    output_path = "static/density_scale_data.json"
    generate_density_scale_data(data_path, output_path)
