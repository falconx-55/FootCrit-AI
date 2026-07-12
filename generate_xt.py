import s3fs
import pandas as pd
import numpy as np

x_zones = 16
y_zones = 12
xt_surface = np.zeros((y_zones, x_zones))
for col in range(x_zones):
    threat_value = (col / (x_zones - 1)) ** 3  
    xt_surface[:, col] = threat_value * 0.5  


def get_xt_value(x, y):
    if pd.isna(x) or pd.isna(y):
        return 0.0

    # StatsBomb pitch is 120x80. Map to 16x12 grid.
    col = min(int((x / 120.0) * x_zones), x_zones - 1)
    row = min(int((y / 80.0) * y_zones), y_zones - 1)

    col = max(0, col)
    row = max(0, row)

    return xt_surface[row, col]


def calculate_deterministic_xt(row):
    """Calculates the true tactical value added by a pass or carry."""
    start_x, start_y = row['start_x'], row['start_y']
    base_xt = get_xt_value(start_x, start_y)


    if pd.notna(row.get('pass.end_location')):
        end_loc = row['pass.end_location']
        if isinstance(end_loc, (list, np.ndarray)) and len(end_loc) == 2:
            end_xt = get_xt_value(end_loc[0], end_loc[1])
            return end_xt - base_xt


    elif pd.notna(row.get('carry.end_location')):
        end_loc = row['carry.end_location']
        if isinstance(end_loc, (list, np.ndarray)) and len(end_loc) == 2:
            end_xt = get_xt_value(end_loc[0], end_loc[1])
            return end_xt - base_xt


    elif row.get('type.name') == 'Shot':
        return 0.5 - base_xt

    return 0.0


def update_parquet_with_xt(s3_folder):
    s3 = s3fs.S3FileSystem(anon=False)
    s3_path = s3_folder.replace("s3://", "").rstrip("/")

    print(f"Scanning {s3_path} for Parquet files...")
    files = [f"s3://{f}" for f in s3.find(s3_path) if f.endswith('.parquet')]
    print(f"Found {len(files)} files to update.\n")

    for i, file_path in enumerate(files):

        with s3.open(file_path, 'rb') as f:
            df = pd.read_parquet(f)


        df['start_x'] = df['location'].apply(
            lambda loc: loc[0] if isinstance(loc, np.ndarray) or isinstance(loc, list) else np.nan)
        df['start_y'] = df['location'].apply(
            lambda loc: loc[1] if isinstance(loc, np.ndarray) or isinstance(loc, list) else np.nan)

   
        df['xt_added'] = df.apply(calculate_deterministic_xt, axis=1)

     
        df = df.drop(columns=['start_x', 'start_y'])


        df.to_parquet(file_path, engine='pyarrow', compression='snappy')
        print(f"[{i + 1}/{len(files)}] Updated & Overwritten: {file_path}")

    print("\nSUCCESS: All Parquet files now contain the deterministic 'xt_added' target tensor!")


if __name__ == "__main__":
    s3_root_url = "s3://sports-ai-datalake-emon/events/"
    update_parquet_with_xt(s3_root_url)
