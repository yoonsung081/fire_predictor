"""
Visualization functions for the fire predictor project.
"""

import json
import webbrowser
import os
import pandas as pd

def export_data_to_json(df, filename, lat_col='LAT', lon_col='LON', extra_cols=None):
    """
    Exports DataFrame data to a JSON file for map visualization.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        filename (str): The name of the output JSON file.
        lat_col (str): The name of the latitude column.
        lon_col (str): The name of the longitude column.
        extra_cols (list): A list of extra columns to include in the properties.
    """
    map_data = []
    for _, row in df.iterrows():
        properties = {}
        if extra_cols:
            for col in extra_cols:
                properties[col] = row[col]
        
        map_data.append({
            'lat': row[lat_col],
            'lon': row[lon_col],
            'properties': properties
        })

    # Ensure the static directory exists
    static_dir = os.path.join(os.path.dirname(__file__), '..', 'static')
    os.makedirs(static_dir, exist_ok=True)

    filepath = os.path.join(static_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(map_data, f, ensure_ascii=False, indent=4)
    print(f"Map data exported to {filepath}")

def open_dashboard():
    """
    Opens the main dashboard (index.html) in a web browser.
    """
    # Correctly construct the path to index.html relative to this script's location
    # __file__ -> src/visualize.py
    # os.path.dirname(__file__) -> src/
    # os.path.join(..., '..') -> project root
    project_root = os.path.join(os.path.dirname(__file__), '..')
    dashboard_path = os.path.join(project_root, 'index.html')
    
    if os.path.exists(dashboard_path):
        webbrowser.open('file://' + os.path.realpath(dashboard_path))
    else:
        print(f"Error: {dashboard_path} not found.")

# The old functions are no longer needed as the new index.html will handle visualization.
# You can keep them for reference or remove them.

# def show_fire_map(center, df, popup_col, color):
#     ...

# def show_long_term_prediction_map(center, pred_dfs):
#     ...