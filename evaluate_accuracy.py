

import json
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of the Earth in kilometers

    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

def evaluate_accuracy(actual_data_path, predicted_data_path, distance_threshold_km=10):
    with open(actual_data_path, 'r', encoding='utf-8') as f:
        actual_fires = json.load(f)

    with open(predicted_data_path, 'r', encoding='utf-8') as f:
        predicted_fires = json.load(f)

    correct_predictions = 0
    evaluated_actual_fires = 0
    
    for actual_fire in actual_fires:
        actual_lat = actual_fire['latitude']
        actual_lon = actual_fire['longitude']
        actual_time_str = actual_fire['start_time']
        actual_time = datetime.strptime(actual_time_str, '%Y-%m-%d %H:%M')

        found_location_and_time_match = False
        has_predicted_date_match = False

        for predicted_fire in predicted_fires:
            predicted_lat = predicted_fire['lat']
            predicted_lon = predicted_fire['lon']
            predicted_time_str = predicted_fire['date']
            predicted_time = datetime.strptime(predicted_time_str, '%Y%m%d')

            # Check time match (only date)
            time_match = (actual_time.year == predicted_time.year and
                          actual_time.month == predicted_time.month and
                          actual_time.day == predicted_time.day)
            
            if time_match:
                has_predicted_date_match = True
                # Check location proximity
                distance = haversine(actual_lat, actual_lon, predicted_lat, predicted_lon)
                location_match = (distance <= distance_threshold_km)

                if location_match:
                    correct_predictions += 1
                    found_location_and_time_match = True
                    break # Move to the next actual fire once a match is found for both time and location
        
        if has_predicted_date_match:
            evaluated_actual_fires += 1

    accuracy = (correct_predictions / evaluated_actual_fires) * 100 if evaluated_actual_fires > 0 else 0
    return accuracy, evaluated_actual_fires, correct_predictions

actual_data_path = r"C:\Users\000\OneDrive\Desktop\fire_predictor_project\fire_data_with_updated_coords.json"
predicted_data_path = r"C:\Users\000\OneDrive\Desktop\fire_predictor_project\data\daily_top_future_predictions.json"

accuracy, total, correct = evaluate_accuracy(actual_data_path, predicted_data_path)

print(f"총 실제 화재 발생 건수: {total}")
print(f"정확히 예측된 화재 건수: {correct}")
print(f"예측 정확도: {accuracy:.2f}%")

