import torch
import torch_geometric
from torch_geometric.explain import Explainer, GNNExplainer
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Import your existing modules
from st_gnn_model import TacticalSTGNN
from enterprise_stream import create_streaming_dataloader


def generate_visual_explanation():
    print("============================================================")
    print("🔍 INITIALIZING GNN EXPLAINER ALGORITHM")
    print("============================================================\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load the trained architecture
    print("1. Loading Tactical ST-GNN...")
    model = TacticalSTGNN(num_node_features=8, hidden_channels=64, rnn_hidden=128)
    model.load_state_dict(torch.load("best_tactical_stgnn.pt", map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # 2. Fetch a high-threat tactical anomaly from AWS S3
    print("2. Fetching validation data...")
    s3_root_url = "s3://your-bucket-name/events/"
    val_loader = create_streaming_dataloader(s3_folder=s3_root_url, batch_size=1, split="val", seq_len=3)

    target_graph = None
    with torch.no_grad():
        for batch_seq in val_loader:
            # For explainability, we target the final spatial frame of the sequence
            target_graph = batch_seq[-1].to(device)
            break  # Grab the first available anomaly for the demonstration

    # 3. Configure the GNNExplainer
    # We define exactly what we want to mask: node attributes (features) and edges (passing lanes)
    print("3. Configuring the GNNExplainer Masking Engine...")
    explainer = Explainer(
        model=model.gat1,  # We target the first attention layer to see spatial focus
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='binary_classification',
            task_level='node',
            return_type='raw',
        ),
    )

    # 4. Generate the Explanation Matrices
    print("4. Optimizing masks to maximize mutual information...")
    explanation = explainer(target_graph.x, target_graph.edge_index)

    # Extract the masks
    edge_mask = explanation.edge_mask.cpu().detach().numpy()
    node_mask = explanation.node_mask.cpu().detach().numpy()

    # Calculate overall node importance by summing feature importance across each node
    node_importance = node_mask.sum(axis=1)

    print("   -> Masks successfully generated!")

    # 5. Render the Tactical Output
    print("5. Rendering visual proof to 'tactical_explanation.png'...")
    render_pitch_map(target_graph, edge_mask, node_importance)
    print("\nSUCCESS: Visual explainability pipeline complete.")


def render_pitch_map(graph, edge_mask, node_importance):
    """Draws the network graph overlaid on a standard 120x80 StatsBomb pitch."""
    G = torch_geometric.utils.to_networkx(graph, to_undirected=False)

    # ---------------------------------------------------------
    # THE FIX: UN-NORMALIZE THE COORDINATES
    # Multiplying by standard pitch dimensions (120m x 80m)
    # ---------------------------------------------------------
    pos = {i: (graph.x[i, 0].item() * 120, graph.x[i, 1].item() * 80) for i in range(graph.num_nodes)}

    plt.figure(figsize=(12, 8))

    # Filter for highly weighted edges
    threshold = np.percentile(edge_mask, 85)
    edges_to_draw = [(u, v) for i, (u, v) in enumerate(G.edges()) if edge_mask[i] >= threshold]

    # Scale edge weights for visual thickness
    max_mask = np.max(edge_mask) if np.max(edge_mask) > 0 else 1.0
    edge_weights = [(edge_mask[i] / max_mask) * 4 + 1 for i, (u, v) in enumerate(G.edges()) if
                    edge_mask[i] >= threshold]

    # Scale node sizes based on importance
    max_imp = np.max(node_importance) if np.max(node_importance) > 0 else 1.0
    node_sizes = [100 + ((imp / max_imp) * 800) for imp in node_importance]

    # Draw elements
    nx.draw_networkx_nodes(G, pos, node_color='crimson', node_size=node_sizes, alpha=0.9, edgecolors='white')
    nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw, edge_color='gold', width=edge_weights, alpha=0.7,
                           arrows=True)
    nx.draw_networkx_labels(G, pos, font_size=9, font_color='white', font_weight='bold')

    plt.title("GNNExplainer: AI Spatial Attention & High-Threat Passing Lanes", fontsize=14, color='white', pad=20)
    plt.gca().set_facecolor('#2b2b2b')
    plt.gcf().patch.set_facecolor('#1e1e1e')

    # Lock the camera strictly to standard football pitch dimensions
    plt.xlim(0, 120)
    plt.ylim(0, 80)

    plt.grid(color='#404040', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("tactical_explanation.png", dpi=300, bbox_inches='tight', facecolor='#1e1e1e')
    plt.close()

if __name__ == "__main__":
    generate_visual_explanation()