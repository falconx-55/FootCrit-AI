import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import boto3
from statsbombpy import sb

s3_client = boto3.client('s3')
BUCKET_NAME = "your-bucket-name"


def process_and_upload(comp_id, season_id):
    matches = sb.matches(competition_id=comp_id, season_id=season_id)

    for match_id in matches['match_id'].unique():
        events = sb.events(match_id=match_id, fmt='dict')

        all_shots = []

        for event in events.values():
            if event.get('type', {}).get('name') == 'Shot':
                if 'shot' in event and 'freeze_frame' in event['shot']:
                    event['match_id'] = match_id
                    event['shooter_name'] = event.get('player', {}).get('name')
                    all_shots.append(event)

        if not all_shots:
            continue

        df_flattened = pd.json_normalize(
            all_shots,
            record_path=['shot', 'freeze_frame'],
            meta=['match_id', 'timestamp', 'shooter_name'],
            errors='ignore'
        )

        schema = pa.schema([
            ('teammate', pa.bool_()),
            ('location', pa.list_(pa.float64())),
            ('player.id', pa.int32()),
            ('player.name', pa.dictionary(pa.int32(), pa.string())),
            ('position.id', pa.int32()),
            ('position.name', pa.dictionary(pa.int32(), pa.string())),
            ('match_id', pa.int32()),
            ('timestamp', pa.string()),
            ('shooter_name', pa.dictionary(pa.int32(), pa.string()))
        ])

        df_flattened = df_flattened.reindex(columns=[field.name for field in schema])

        table = pa.Table.from_pandas(df_flattened, schema=schema)

        local_file = f"match_{match_id}.parquet"
        pq.write_table(table, local_file, compression='snappy')

        s3_key = f"events/competition_id={comp_id}/season_id={season_id}/match_{match_id}.parquet"
        s3_client.upload_file(local_file, BUCKET_NAME, s3_key)

        os.remove(local_file)
        print(f"Successfully uploaded {s3_key}")


if __name__ == "__main__":
    process_and_upload(comp_id=9, season_id=281)