from geopy.geocoders import Nominatim
import time, json

# ===== 설정 =====
INPUT_FILE = "fire_data_total.json"     # 기존 JSON 파일
OUTPUT_FILE = "fire_data_with_coords.json"
# =================

# 파일 로드
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# geocoder 설정
geolocator = Nominatim(user_agent="fire_location_converter")

converted = []
for i, row in enumerate(data, start=1):
    try:
        start, end, location, status, step, etc = row
        print(f"📍 [{i}] {location} 변환 중...")

        # 주소를 좌표로 변환
        geo = geolocator.geocode(location)
        if geo:
            lat, lon = geo.latitude, geo.longitude
        else:
            lat, lon = None, None

        converted.append({
            "start_time": start,
            "end_time": end,
            "location": location,
            "latitude": lat,
            "longitude": lon,
            "status": status,
            "step": step,
            "etc": etc
        })

        # API 사용 제한 대비 딜레이
        time.sleep(1)

    except Exception as e:
        print(f"⚠️ [{i}] 변환 실패: {e}")

# 결과 저장
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(converted, f, ensure_ascii=False, indent=2)

print(f"\n🔥 총 {len(converted)}건 변환 완료 → {OUTPUT_FILE}")
