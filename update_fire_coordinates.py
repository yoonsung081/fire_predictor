import json
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of Earth (km)
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def find_nearest_peak(fire_coord, peaks_data):
    nearest_peak = None
    min_distance = float('inf')

    for peak in peaks_data['features']:
        peak_coords = peak['geometry']['coordinates']
        distance = haversine(fire_coord['latitude'], fire_coord['longitude'],
                             peak_coords[1], peak_coords[0])
        if distance < min_distance:
            min_distance = distance
            nearest_peak = peak
    return nearest_peak

with open(r"C:\Users\000\OneDrive\Desktop\fire_predictor_project\fire_data_with_coords.json", 'r', encoding='utf-8') as f:
    fire_data = json.load(f)

with open(r"C:\Users\000\OneDrive\Desktop\fire_predictor_project\data\peaks_cache.geojson", 'r', encoding='utf-8') as f:
    peaks_data = json.load(f)

updated_fire_data = []

for fire_incident in fire_data:
    fire_coord = {
        'latitude': fire_incident['latitude'],
        'longitude': fire_incident['longitude']
    }

    nearest_peak = find_nearest_peak(fire_coord, peaks_data)
    if nearest_peak:
        peak_coords = nearest_peak['geometry']['coordinates']
        fire_incident['latitude'] = peak_coords[1]
        fire_incident['longitude'] = peak_coords[0]

    updated_fire_data.append(fire_incident)

with open(r"C:\Users\000\OneDrive\Desktop\fire_predictor_project\fire_data_with_updated_coords.json", 'w', encoding='utf-8') as f:
    json.dump(updated_fire_data, f, indent=2, ensure_ascii=False)

print("✅ 좌표 업데이트가 완료되었습니다. fire_data_with_updated_coords.json 파일을 확인하세요.")
