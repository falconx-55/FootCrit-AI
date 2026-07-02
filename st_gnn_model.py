import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class TacticalSTGNN(nn.Module):
    def __init__(self, num_node_features=8, hidden_channels=64, rnn_hidden=128):
        super().__init__()

        # 1. The Spatial Engine
        self.gat1 = GATConv(num_node_features, hidden_channels)
        self.gat2 = GATConv(hidden_channels, hidden_channels)

        # 2. The Temporal Engine
        self.gru = nn.GRU(input_size=hidden_channels, hidden_size=rnn_hidden, batch_first=True)

        # 3. The Final Threat Predictor (UPGRADED: The Reasoning Engine)
        self.predictor = nn.Sequential(
            nn.Linear(rnn_hidden, 64),
            nn.GELU(),  # Non-linear reasoning
            nn.Dropout(0.2),  # Forces the network to rely on multiple tactical patterns
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Linear(16, 1)  # Final xT prediction
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.gat1(x, edge_index, edge_attr))
        x = F.relu(self.gat2(x, edge_index, edge_attr))
        x = global_mean_pool(x, batch)

        x_seq = x.unsqueeze(1)
        gru_out, hidden_state = self.gru(x_seq)

        # Pass the memory through the new Reasoning Engine
        return self.predictor(hidden_state[-1])