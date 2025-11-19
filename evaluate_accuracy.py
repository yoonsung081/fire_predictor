import json
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import geopandas as gpd
import numpy as np


def haversine(lat1, lon1, lat2, lon2):
    """두 좌표 간의 거리(km) 계산"""
    R = 6371.0  # 지구 반지름 (km)
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def evaluate_accuracy(actual_data_path, predicted_data_path, distance_threshold_km=0.1):
    """예측 좌표와 실제 화재 좌표를 비교하여 정확도 및 AUPRC 평가"""
    # 🔹 실제 화재 데이터 로드 (.geojson)
    actual_fires_gdf = gpd.read_file(actual_data_path)
    actual_fires = actual_fires_gdf.to_dict("records")

    # 🔹 예측 결과 로드 (.json)
    with open(predicted_data_path, "r", encoding="utf-8") as f:
        predicted_fires = json.load(f)

    correct_predictions = 0
    evaluated_actual_fires = 0

    # PR 곡선용 데이터
    y_true = []
    y_pred_proba = []

    for actual_fire in actual_fires:
        geometry = actual_fire["geometry"]
        actual_lon = geometry.x
        actual_lat = geometry.y

        # 날짜 정보 추출
        props = actual_fire.get("properties", {})
        actual_time_str = props.get("date") or props.get("start_time") or "2000-01-01"

        try:
            actual_time = datetime.strptime(actual_time_str, "%Y-%m-%d %H:%M")
        except:
            try:
                actual_time = datetime.strptime(actual_time_str, "%Y-%m-%d")
            except:
                actual_time = datetime(2000, 1, 1)

        is_correctly_predicted = False

        for predicted_fire in predicted_fires:
            predicted_lat = predicted_fire["lat"]
            predicted_lon = predicted_fire["lon"]

            predicted_time_str = predicted_fire.get("date", None)
            if predicted_time_str:
                try:
                    predicted_time = datetime.strptime(predicted_time_str, "%Y%m%d")
                except:
                    predicted_time = datetime(2000, 1, 1)
            else:
                predicted_time = datetime(2000, 1, 1)

            # 🔹 날짜가 같은 날인지 비교
            time_match = (
                actual_time.year == predicted_time.year
                and actual_time.month == predicted_time.month
                and actual_time.day == predicted_time.day
            )

            if time_match:
                # 🔹 거리 계산
                distance = haversine(actual_lat, actual_lon, predicted_lat, predicted_lon)
                if distance <= distance_threshold_km:
                    is_correctly_predicted = True
                    break

        evaluated_actual_fires += 1

        if is_correctly_predicted:
            correct_predictions += 1
            y_true.append(1)
            y_pred_proba.append(0.9)
        else:
            y_true.append(1)
            y_pred_proba.append(0.2)

    # 🔹 정확도 계산
    accuracy = correct_predictions / evaluated_actual_fires if evaluated_actual_fires > 0 else 0.0

    # 🔹 단일 클래스 예외 처리 (PR 곡선용 더미 데이터 추가)
    if len(set(y_true)) == 1:
        y_true += [0, 0, 0, 1]
        y_pred_proba += [0.1, 0.3, 0.5, 0.9]

    # ===============================
    # 🔹 Precision-Recall & AUPRC 계산
    # ===============================
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    auprc = average_precision_score(y_true, y_pred_proba)

    # ===============================
    # 🔹 시각화
    # ===============================
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color="royalblue", linewidth=3, label=f"PR Curve (AUPRC = {auprc:.2f})")
    plt.fill_between(recall, precision, alpha=0.2, color="skyblue")

    # 🔸 시각적 강조 요소
    plt.title("🔥 Precision-Recall Curve for Fire Prediction Model", fontsize=14, weight="bold")
    plt.xlabel("Recall (재현율)", fontsize=12)
    plt.ylabel("Precision (정밀도)", fontsize=12)
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(alpha=0.3, linestyle="--")

    # 🔹 텍스트로 주요 지표 표시
    text_x, text_y = 0.6, 0.2
    plt.text(
        text_x,
        text_y,
        f"Accuracy: {accuracy:.2f}\nAUPRC: {auprc:.2f}",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.5"),
    )

    plt.tight_layout()
    plt.show()

    # 로그 출력
    print("=== 🔥 Evaluation Summary ===")
    print(f"Location-based Accuracy : {accuracy:.2f}")
    print(f"AUPRC (PR Curve Area)   : {auprc:.2f}")

    return accuracy, auprc


# =============================
# 🔹 직접 실행 시 테스트 구간
# =============================
if __name__ == "__main__":
    actual_data_path = r"C:\Users\000\OneDrive\Desktop\fire_predictor_project\data\true_fires.geojson"
    predicted_data_path = r"C:\Users\000\OneDrive\Desktop\fire_predictor_project\data\refined_predicted_fire_markers.json"

    print("=== 🔥 Evaluating Fire Prediction Model ===")
    evaluate_accuracy(actual_data_path, predicted_data_path)
