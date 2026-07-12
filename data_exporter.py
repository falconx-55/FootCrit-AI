import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class TacticalDataExporter:
    @staticmethod
    def export_to_parquet(tracking_data, output_path):
        if not tracking_data:
            return None

        df = pd.DataFrame(tracking_data)

        df['frame'] = df['frame'].astype(int)
        df['track_id'] = df['track_id'].astype(int)
        df['class_id'] = df['class_id'].astype(int)
        df['pitch_x'] = df['pitch_x'].astype(float)
        df['pitch_y'] = df['pitch_y'].astype(float)

        schema = pa.schema([
            ('frame', pa.int64()),
            ('track_id', pa.int64()),
            ('class_id', pa.int64()),
            ('pitch_x', pa.float64()),
            ('pitch_y', pa.float64())
        ])

        table = pa.Table.from_pandas(df, schema=schema)
        pq.write_table(table, output_path)

        print(f"Parquet serialization complete. Target: {output_path}")
        return output_path