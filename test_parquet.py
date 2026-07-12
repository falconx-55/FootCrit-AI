import pandas as pd
import os
import tempfile


def verify_parquet_pipeline():
    parquet_path = os.path.join(tempfile.gettempdir(), "tactical_telemetry.parquet")

    if not os.path.exists(parquet_path):
        print("ERROR: Parquet file not found. Run the Streamlit pipeline first.")
        return

    print(f"LOADING PARQUET FILE: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    print("\n--- DATAFRAME SCHEMA ---")
    print(df.dtypes)

    print("\n--- DATAFRAME SHAPE ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\n--- DATA SAMPLE (FIRST 5 ROWS) ---")
    print(df.head())

    print("\nPIPELINE VERIFICATION COMPLETE.")


if __name__ == "__main__":
    verify_parquet_pipeline()