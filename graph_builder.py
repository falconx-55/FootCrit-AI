import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np
import math


class TacticalGraphBuilder:
    def __init__(self, distance_threshold=25.0):
        # Matches the 25.0 threshold from your enterprise_stream.py
        self.distance_threshold = distance_threshold

    def build_graphs_from_parquet(self, parquet_path):
        df = pd.read_parquet(parquet_path)
        graphs = []

        for frame_id, frame_data in df.groupby('frame'):
            players = frame_data[frame_data['class_id'] == 0]
            balls = frame_data[frame_data['class_id'] == 32]

            # Locate the ball for heuristic calculations
            ball_x, ball_y = None, None
            if not balls.empty:
                ball_x = balls.iloc[0]['pitch_x']
                ball_y = balls.iloc[0]['pitch_y']

            node_features = []
            coords = []

            for _, row in players.iterrows():
                x = row['pitch_x']
                y = row['pitch_y']
                coords.append([x, y])

                # 1 & 2: Normalized Coordinates
                norm_x = x / 120.0
                norm_y = y / 80.0

                # 3. Teammate Inference (Mocked as 0.5 unknown for pure CV without color clustering)
                is_teammate = 0.5

                # 4. Position ID (Mocked as unknown for pure CV)
                pos_id = 0.0

                # 5. Ball Carrier Inference (Closest player to the ball < 2 meters)
                is_carrier = 0.0
                if ball_x is not None and ball_y is not None:
                    dist_to_ball = math.hypot(x - ball_x, y - ball_y)
                    if dist_to_ball < 2.0:
                        is_carrier = 1.0

                # 6. Pressure Norm (Players within 4 meters)
                pressure = 0.0
                for _, other in players.iterrows():
                    if row['track_id'] != other['track_id']:
                        dist = math.hypot(x - other['pitch_x'], y - other['pitch_y'])
                        if dist < 4.0:
                            pressure += 1.0
                pressure_norm = min(pressure / 5.0, 1.0)

                # 7 & 8. Goal Geometry
                dist_to_goal = math.hypot(120 - x, 40 - y) / 140.0
                angle_to_goal = math.atan2(40 - y, 120 - x) / math.pi

                node_features.append([
                    norm_x, norm_y, is_teammate, pos_id,
                    is_carrier, pressure_norm, dist_to_goal, angle_to_goal
                ])

            if len(node_features) < 2:
                continue

            # Convert to Tensors
            x_tensor = torch.tensor(node_features, dtype=torch.float)

            # Build Edges based on distance matrix
            num_nodes = len(node_features)
            edge_indices = []
            edge_attrs = []

            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        dist = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
                        if dist < self.distance_threshold:
                            edge_indices.append([i, j])
                            edge_attrs.append([dist])

            if len(edge_indices) > 0:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 1), dtype=torch.float)

            # CV data has no true 'y' target label, mocked to 0.0 for inference
            y_tensor = torch.tensor([[0.0]], dtype=torch.float)

            graph = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor)
            graphs.append(graph)

        print(f"Graph Construction Complete. Generated {len(graphs)} 8-Dimensional sequential tactical frames.")
        return graphs


if __name__ == "__main__":
    builder = TacticalGraphBuilder()
    graph_sequence = builder.build_graphs_from_parquet("tactical_telemetry.parquet")

    if len(graph_sequence) > 0:
        sample_graph = graph_sequence[0]
        print(f"Sample Node Tensor Shape: {sample_graph.x.shape} (Must be [N, 8])")
        print(f"Sample Edge Index Shape: {sample_graph.edge_index.shape}")