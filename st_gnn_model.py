import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class TacticalSTGNN(nn.Module):
    def __init__(self, num_node_features=8, hidden_channels=64, rnn_hidden=128):
        super().__init__()

  
        self.gat1 = GATConv(num_node_features, hidden_channels)
        self.gat2 = GATConv(hidden_channels, hidden_channels)

   
        self.gru = nn.GRU(input_size=hidden_channels, hidden_size=rnn_hidden, batch_first=True)

      
        self.predictor = nn.Sequential(
            nn.Linear(rnn_hidden, 64),
            nn.GELU(),  
            nn.Dropout(0.2), 
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Linear(16, 1)  
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.gat1(x, edge_index, edge_attr))
        x = F.relu(self.gat2(x, edge_index, edge_attr))
        x = global_mean_pool(x, batch)

        x_seq = x.unsqueeze(1)
        gru_out, hidden_state = self.gru(x_seq)

       
        return self.predictor(hidden_state[-1])
