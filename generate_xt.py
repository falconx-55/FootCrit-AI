import s3fs
import pandas as pd
import numpy as np

# Create a 16x12 Expected Threat (xT) surface
# Threat increases exponentially as X approaches 120 (the opponent's goal)
x_zones = 16
y_zones = 12
xt_surface = np.zeros((y_zones, x_zones))
for col in range(x_zones):
    threat_value = (col / (x_zones - 1)) ** 3  # Exponential increase
    xt_surface[:, col] = threat_value * 0.5  # Max base threat of 0.5


def get_xt_value(x, y):
    if pd.isna(x) or pd.isna(y):
        return 0.0

    # StatsBomb pitch is 120x80. Map to 16x12 grid.
    col = min(int((x / 120.0) * x_zones), x_zones - 1)
    row = min(int((y / 80.0) * y_zones), y_zones - 1)

    # Ensure coordinates don't drop below 0
    col = max(0, col)
    row = max(0, row)

    return xt_surface[row, col]


def calculate_deterministic_xt(row):
    """Calculates the true tactical value added by a pass or carry."""
    start_x, start_y = row['start_x'], row['start_y']
    base_xt = get_xt_value(start_x, start_y)

    # 1. If the event is a pass with an end location
    if pd.notna(row.get('pass.end_location')):
        end_loc = row['pass.end_location']
        if isinstance(end_loc, (list, np.ndarray)) and len(end_loc) == 2:
            end_xt = get_xt_value(end_loc[0], end_loc[1])
            return end_xt - base_xt

    # 2. If the event is a carry with an end location
    elif pd.notna(row.get('carry.end_location')):
        end_loc = row['carry.end_location']
        if isinstance(end_loc, (list, np.ndarray)) and len(end_loc) == 2:
            end_xt = get_xt_value(end_loc[0], end_loc[1])
            return end_xt - base_xt

    # 3. If it's a shot, the threat added is the probability of a goal (simplified here to max threat)
    elif row.get('type.name') == 'Shot':
        return 0.5 - base_xt

        # 4. If the ball didn't move meaningfully (e.g., a foul, substitution, or static event)
    return 0.0


def update_parquet_with_xt(s3_folder):
    s3 = s3fs.S3FileSystem(anon=False)
    s3_path = s3_folder.replace("s3://", "").rstrip("/")

    print(f"Scanning {s3_path} for Parquet files...")
    files = [f"s3://{f}" for f in s3.find(s3_path) if f.endswith('.parquet')]
    print(f"Found {len(files)} files to update.\n")

    for i, file_path in enumerate(files):
        # 1. Download into RAM using the "byte stream" blindfold trick
        with s3.open(file_path, 'rb') as f:
            df = pd.read_parquet(f)

        # 2. Extract X, Y coordinates for origin
        df['start_x'] = df['location'].apply(
            lambda loc: loc[0] if isinstance(loc, np.ndarray) or isinstance(loc, list) else np.nan)
        df['start_y'] = df['location'].apply(
            lambda loc: loc[1] if isinstance(loc, np.ndarray) or isinstance(loc, list) else np.nan)

        # 3. Calculate Deterministic xT Added based on physical ball movement
        df['xt_added'] = df.apply(calculate_deterministic_xt, axis=1)

        # Drop temporary columns to keep Parquet clean
        df = df.drop(columns=['start_x', 'start_y'])

        # 4. Upload back to AWS, overwriting the old file
        df.to_parquet(file_path, engine='pyarrow', compression='snappy')
        print(f"[{i + 1}/{len(files)}] Updated & Overwritten: {file_path}")

    print("\nSUCCESS: All Parquet files now contain the deterministic 'xt_added' target tensor!")


if __name__ == "__main__":
    s3_root_url = "s3://your-bucket-name/events/"
    update_parquet_with_xt(s3_root_url)