import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
import numpy as np
import json
import os

# --- Configuration ---
PREDICTED_JSON_PATH = "data/predicted_fire_markers.json"
PEAKS_GEOJSON_PATH = "data/peaks_cache.geojson"
ACTUAL_FIRES_PATH = "data/fixed_actual.csv"
REFINED_PREDICTED_JSON_PATH = "data/refined_predicted_fire_markers.json"
REFINED_PREDICTED_CSV_PATH = "data/fixed_predict.csv" # Update this for map visualization

# Overlap threshold in degrees (approx. 0.01 degree ~ 1 km at equator)
OVERLAP_THRESHOLD = 0.1 # Used to detect overlap
ADJUSTMENT_OFFSET = 0.005 # Degrees to move overlapping markers

def find_nearest_peak(predicted_points, peak_tree, peak_gdf):
    """Finds the nearest peak for each predicted fire location."""
    # Query the KD-tree for the nearest peak to each predicted point
    distances, indices = peak_tree.query(predicted_points)
    
    # Get the nearest peak coordinates
    nearest_peaks = peak_gdf.iloc[indices]
    return nearest_peaks.geometry.x, nearest_peaks.geometry.y # x is longitude, y is latitude

def adjust_overlapping_predictions(predicted_df, actual_df, threshold, offset):
    """Adjusts predicted fire locations that overlap with actual fire locations."""
    if actual_df.empty:
        return predicted_df

    # Create KD-tree for actual fire locations
    actual_points = np.array(list(zip(actual_df['LON'], actual_df['LAT'])))
    actual_tree = cKDTree(actual_points)

    # Prepare predicted points for querying
    predicted_points = np.array(list(zip(predicted_df['lon'], predicted_df['lat'])))

    # Find distances from each predicted point to its nearest actual fire point
    distances, _ = actual_tree.query(predicted_points)

    # Identify overlapping predictions
    overlapping_indices = np.where(distances <= threshold)[0]

    # Apply adjustment to overlapping predictions
    for i, idx in enumerate(overlapping_indices):
        # Simple alternating offset to spread them out
        if i % 4 == 0: # Move slightly north
            predicted_df.loc[idx, 'lat'] += offset
        elif i % 4 == 1: # Move slightly east
            predicted_df.loc[idx, 'lon'] += offset
        elif i % 4 == 2: # Move slightly south
            predicted_df.loc[idx, 'lat'] -= offset
        else: # Move slightly west
            predicted_df.loc[idx, 'lon'] -= offset
            
    return predicted_df

if __name__ == "__main__":
    print("Loading predicted fire markers...")
    if not os.path.exists(PREDICTED_JSON_PATH):
        raise FileNotFoundError(f"Predicted fire markers not found: {PREDICTED_JSON_PATH}. Please run predict_fire_locations.py first.")
    with open(PREDICTED_JSON_PATH, 'r', encoding='utf-8') as f:
        predicted_markers = json.load(f)

    if not predicted_markers:
        print("No predicted fire markers to refine.")
        # Create empty files to avoid errors in subsequent steps
        os.makedirs(os.path.dirname(REFINED_PREDICTED_JSON_PATH), exist_ok=True)
        with open(REFINED_PREDICTED_JSON_PATH, 'w', encoding='utf-8') as f: json.dump([], f)
        os.makedirs(os.path.dirname(REFINED_PREDICTED_CSV_PATH), exist_ok=True)
        pd.DataFrame(columns=['LAT', 'LON', 'FIRE_PROBABILITY']).to_csv(REFINED_PREDICTED_CSV_PATH, index=False)
        exit()

    # Convert predicted markers to a DataFrame for easier processing
    predicted_df = pd.DataFrame(predicted_markers)
    predicted_points_for_peak_search = np.array(list(zip(predicted_df['lon'], predicted_df['lat'])))

    print("Loading mountain peaks data...")
    if not os.path.exists(PEAKS_GEOJSON_PATH):
        raise FileNotFoundError(f"Mountain peaks data not found: {PEAKS_GEOJSON_PATH}.")
    peak_gdf = gpd.read_file(PEAKS_GEOJSON_PATH)
    
    # Create a KD-tree for efficient nearest neighbor search on peaks
    peak_coords = np.array(list(zip(peak_gdf.geometry.x, peak_gdf.geometry.y)))
    peak_tree = cKDTree(peak_coords)

    print("Refining predicted fire locations to nearest peaks...")
    nearest_peak_lon, nearest_peak_lat = find_nearest_peak(predicted_points_for_peak_search, peak_tree, peak_gdf)

    # Update predicted locations with nearest peak coordinates
    predicted_df['lon'] = nearest_peak_lon.values
    predicted_df['lat'] = nearest_peak_lat.values

    print("Loading actual fire locations for overlap check...")
    if not os.path.exists(ACTUAL_FIRES_PATH):
        print(f"WARNING: Actual fires data not found at {ACTUAL_FIRES_PATH}. Skipping overlap adjustment.")
        actual_fires_df = pd.DataFrame(columns=['LAT', 'LON'])
    else:
        actual_fires_df = pd.read_csv(ACTUAL_FIRES_PATH)

    print("Adjusting predictions overlapping with actual fires...")
    # Use the original predicted_df for adjustment, as we want to adjust all overlapping ones
    adjusted_predicted_df = adjust_overlapping_predictions(predicted_df, actual_fires_df, OVERLAP_THRESHOLD, ADJUSTMENT_OFFSET)

    # --- Save refined and adjusted markers to JSON ---
    os.makedirs(os.path.dirname(REFINED_PREDICTED_JSON_PATH), exist_ok=True)
    adjusted_predicted_df.to_json(REFINED_PREDICTED_JSON_PATH, orient='records', indent=4, force_ascii=False)
    print(f"Refined and adjusted predicted fire markers saved to {REFINED_PREDICTED_JSON_PATH}")

    # --- Save refined and adjusted markers to CSV for map visualization ---
    os.makedirs(os.path.dirname(REFINED_PREDICTED_CSV_PATH), exist_ok=True)
    adjusted_predicted_df.rename(columns={'lat': 'LAT', 'lon': 'LON'}).to_csv(REFINED_PREDICTED_CSV_PATH, index=False)
    print(f"Refined and adjusted predicted fire locations saved to {REFINED_PREDICTED_CSV_PATH}")

    print("Refinement and adjustment process complete.")
