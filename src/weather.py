import requests

def get_weather(lat, lon, date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
        "timezone": "Asia/Seoul"
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"⚠️ Weather API Error {response.status_code} for {lat}, {lon}, {date}")
        return None

    data = response.json()
    if "daily" not in data or len(data["daily"]["temperature_2m_max"]) == 0:
        print(f"⚠️ No weather data for {lat}, {lon}, {date}")
        return None

    weather = {
        "temp_max": data["daily"]["temperature_2m_max"][0],
        "temp_min": data["daily"]["temperature_2m_min"][0],
        "precip": data["daily"]["precipitation_sum"][0],
        "wind": data["daily"]["windspeed_10m_max"][0]
    }
    return weather
