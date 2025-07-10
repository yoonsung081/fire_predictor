from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import pandas as pd

def add_lat_lon_from_address(df):
    geolocator = Nominatim(user_agent="fire_predictor")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    latitudes = []
    longitudes = []

    for idx, row in df.iterrows():
        address = f"대한민국 {row['발생장소_시도']} {row['발생장소_시군구']} {row['발생장소_읍면']}"
        try:
            location = geocode(address)
            if location:
                latitudes.append(location.latitude)
                longitudes.append(location.longitude)
            else:
                latitudes.append(None)
                longitudes.append(None)
        except:
            latitudes.append(None)
            longitudes.append(None)

    df['LAT'] = latitudes
    df['LON'] = longitudes
    return df
