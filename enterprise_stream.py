import s3fs
import pandas as pd
import torch
from torch.utils.data import IterableDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from itertools import permutations
import math
import random


class StreamingStatsBombDataset(IterableDataset):
    def __init__(self, s3_folder, split="train", val_ratio=0.2, seq_len=3):
        super().__init__()
        self.s3 = s3fs.S3FileSystem(anon=False)
        self.seq_len = seq_len  # Number of chronological events to pass at once

        s3_path = s3_folder.replace("s3://", "").rstrip("/")
        raw_files = self.s3.find(s3_path)

        all_files = sorted([f"s3://{f}" for f in raw_files if f.endswith('.parquet')])
        split_idx = int(len(all_files) * (1 - val_ratio))

        if split == "train":
            self.files = all_files[:split_idx]
            print(f"TRAIN SET: Discovered {len(self.files)} Parquet files ready for streaming.")
        else:
            self.files = all_files[split_idx:]
            print(f"VALIDATION SET: Discovered {len(self.files)} Parquet files ready for streaming.")

    def _build_graph(self, group):
        """Helper function to build a single spatial graph with masked attention."""
        node_features = []
        players = group.to_dict('records')

        for row in players:
            # NORMALIZATION: Squash coordinates between 0 and 1
            x = row['location'][0] / 120.0
            y = row['location'][1] / 80.0

            is_teammate = 1.0 if row['teammate'] else 0.0

            # Position IDs range up to ~25, so divide by 25
            raw_pos = float(row.get('position.id', 0.0)) if pd.notna(row.get('position.id')) else 0.0
            pos_id = raw_pos / 25.0

            is_carrier = 1.0 if row.get('player.name') == row.get('shooter_name') else 0.0

            pressure = 0.0
            for other in players:
                if other['teammate'] != row['teammate']:
                    ox = other['location'][0] / 120.0
                    oy = other['location'][1] / 80.0
                    dist = math.hypot(x - ox, y - oy)
                    # 5 yards is roughly 0.04 in our normalized 0-1 pitch scale
                    if dist < 0.04:
                        pressure += 1.0

            # Pressure maxes out around 5 players, squash to 0-1
            pressure_norm = min(pressure / 5.0, 1.0)

            # Normalize distance to goal (max distance is ~140 yards)
            raw_dist_to_goal = math.hypot(120 - row['location'][0], 40 - row['location'][1])
            dist_to_goal = raw_dist_to_goal / 140.0

            # Angle is naturally between -pi and pi, so we leave it alone or divide by pi
            angle_to_goal = math.atan2(40 - row['location'][1], 120 - row['location'][0]) / math.pi

            node_features.append([
                x, y, is_teammate, pos_id,
                is_carrier, pressure_norm, dist_to_goal, angle_to_goal
            ])
        if len(node_features) < 2:
            return None

        x_tensor = torch.tensor(node_features, dtype=torch.float)
        edge_indices = list(permutations(range(len(node_features)), 2))

        edge_attrs = []
        valid_edge_indices = []

        # MASKED ATTENTION: Only connect players within 25 yards
        for src, dst in edge_indices:
            dist = math.hypot(node_features[src][0] - node_features[dst][0],
                              node_features[src][1] - node_features[dst][1])
            if dist <= 25.0:
                valid_edge_indices.append((src, dst))
                edge_attrs.append([dist])

        if not valid_edge_indices:
            return None

        edge_index = torch.tensor(valid_edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        xt_val = float(group['xt_added'].iloc[0]) if 'xt_added' in group.columns else 0.0
        y_tensor = torch.tensor([[xt_val]], dtype=torch.float)

        return Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor)

    def process_file(self, file_path):
        with self.s3.open(file_path, 'rb') as f:
            df = pd.read_parquet(f)

        # 1. Sort the file by timestamp to guarantee chronological order
        df = df.sort_values(by=['match_id', 'timestamp'])

        # 2. Group into individual events
        grouped_events = list(df.groupby(['match_id', 'timestamp'], sort=False))

        # 3. Create overlapping sliding windows of time (e.g., length 3)
        for i in range(len(grouped_events) - self.seq_len + 1):
            time_window = grouped_events[i: i + self.seq_len]

            graph_sequence = []
            for _, group in time_window:
                graph = self._build_graph(group)
                if graph is not None:
                    graph_sequence.append(graph)

            # 4. Only yield if all 3 frames successfully built a graph
            if len(graph_sequence) == self.seq_len:

                # --- DATA BALANCING (UNDERSAMPLING THE MAJORITY CLASS) ---
                # Check the true xT target of the final frame
                final_target = graph_sequence[-1].y.item()

                # If the pass is "boring" (between -0.01 and 0.01)
                if -0.01 < final_target < 0.01:
                    # 85% of the time, we drop it so the AI isn't overwhelmed by midfield passes
                    if random.random() < 0.85:
                        continue

                        # If it's a dangerous attack, a severe turnover, or the 15% of surviving boring passes:
                yield graph_sequence

    def __iter__(self):
        if not self.files:
            return
        for file_path in self.files:
            yield from self.process_file(file_path)


def create_streaming_dataloader(s3_folder, batch_size=32, split="train", seq_len=3):
    dataset = StreamingStatsBombDataset(s3_folder, split=split, seq_len=seq_len)
    return DataLoader(dataset, batch_size=batch_size)


if __name__ == "__main__":
    s3_root_url = "s3://your-bucket-name/events/"
    dataloader = create_streaming_dataloader(s3_folder=s3_root_url, batch_size=2, split="train")

    for batch_seq in dataloader:
        print("\nSUCCESS! Temporal Data pipeline test passed.")
        print(f"Sequence Length: {len(batch_seq)} time steps")
        print(f"Data type of Step 1: {type(batch_seq[0])}")
        print(f"Target value of Final Step: {batch_seq[-1].y}")
        break