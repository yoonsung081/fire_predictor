
import pandas as pd
import numpy as np
import os

# Configuration
INPUT_CSV_PATH = "data/rf_predict.csv"
OUTPUT_CSV_PATH = "data/rf_predict.csv" # Overwrite the original file
OFFSET_MAGNITUDE = 0.001 # Degrees, approx 111 meters at the equator

if __name__ == "__main__":
    if not os.path.exists(INPUT_CSV_PATH):
        raise FileNotFoundError(f"File not found: {INPUT_CSV_PATH}")

    print(f"Loading {INPUT_CSV_PATH}...")
    df = pd.read_csv(INPUT_CSV_PATH)

    # Generate random offsets
    num_rows = len(df)
    lat_offsets = (np.random.rand(num_rows) - 0.5) * 2 * OFFSET_MAGNITUDE
    lon_offsets = (np.random.rand(num_rows) - 0.5) * 2 * OFFSET_MAGNITUDE

    # Apply offsets
    df['LAT'] += lat_offsets
    df['LON'] += lon_offsets

    print(f"Saving modified data to {OUTPUT_CSV_PATH}...")
    df.to_csv(OUTPUT_CSV_PATH, index=False)

    print("Adjustment complete.")
