import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# dummy 예제 (너의 실제 예측 결과로 교체하면 됨)
y_true = np.random.choice([0, 1], size=300, p=[0.7, 0.3])
y_pred_weather = np.random.choice([0, 1], size=300, p=[0.2, 0.8])
y_pred_baseline = np.random.choice([0, 1], size=300, p=[0.4, 0.6])

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "precision": round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
        "f1": round(f1_score(y_true, y_pred, zero_division=0) * 100, 2)
    }

metrics = {
    "weather": compute_metrics(y_true, y_pred_weather),
    "baseline": compute_metrics(y_true, y_pred_baseline)
}

with open("static/metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print("✅ metrics.json 저장 완료")
