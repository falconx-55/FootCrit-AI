import torch
import torch_geometric
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import CaptumExplainer
from st_gnn_model import TacticalSTGNN
from enterprise_stream import create_streaming_dataloader


def generate_feature_attribution():
    print("============================================================")
    print("INITIALIZING INTEGRATED GRADIENTS ENGINE")
    print("============================================================\n")

    device = torch.device('cpu')

    print("1. Loading Tactical ST-GNN into System RAM...")
    model = TacticalSTGNN(num_node_features=8, hidden_channels=64, rnn_hidden=128)
    model.load_state_dict(torch.load("best_tactical_stgnn.pt", map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    print("2. Fetching validation data...")
    s3_root_url = "s3://your-bucket-name/events/"
    val_loader = create_streaming_dataloader(s3_folder=s3_root_url, batch_size=1, split="val", seq_len=3)

    target_graph = None
    with torch.no_grad():
        for batch_seq in val_loader:
            target_graph = batch_seq[-1].to(device)
            break

    print("3. Configuring the Integrated Gradients Explainer...")
    explainer = Explainer(
        model=model,
        algorithm=CaptumExplainer('IntegratedGradients', internal_batch_size=5),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type=None,
        model_config=dict(
            mode='binary_classification',
            task_level='node',
            return_type='probs',
        ),
    )

    print("4. Calculating gradient integrals across feature dimensions...")
    explanation = explainer(
        target_graph.x,
        target_graph.edge_index,
        edge_attr=target_graph.edge_attr,
        batch=target_graph.batch
    )

    feature_importance = explanation.node_mask.abs().mean(dim=0).cpu().detach().numpy()
    feature_importance = feature_importance / np.sum(feature_importance)

    print("   -> Attribution scores successfully computed!")

    print("5. Rendering mathematical proof to 'feature_attribution.png'...")
    render_attribution_chart(feature_importance)
    print("\nSUCCESS: Feature-level explainability pipeline complete.")


def render_attribution_chart(importance_scores):
    feature_names = [
        "X Coordinate",
        "Y Coordinate",
        "Velocity Magnitude",
        "Velocity Angle",
        "Acceleration",
        "Distance to Ball",
        "Pitch Control Density",
        "Expected Threat (xT)"
    ]

    sorted_indices = np.argsort(importance_scores)
    sorted_scores = importance_scores[sorted_indices]
    sorted_features = [feature_names[i] for i in sorted_indices]

    plt.figure(figsize=(10, 6))

    bars = plt.barh(sorted_features, sorted_scores, color='#ff4757', edgecolor='white', height=0.6)

    plt.title("Integrated Gradients: Feature Attribution for Tactical Anomaly", fontsize=14, color='white', pad=20)
    plt.xlabel("Relative Attribution Weight (Normalized Integral)", fontsize=11, color='#dcdde1')

    plt.gca().set_facecolor('#2f3542')
    plt.gcf().patch.set_facecolor('#1e272e')

    plt.xticks(color='#dcdde1')
    plt.yticks(color='white', fontsize=11)

    plt.grid(axis='x', color='#57606f', linestyle='--', linewidth=0.5, alpha=0.7)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{width * 100:.1f}%',
                 ha='left', va='center', color='white', fontsize=10, fontweight='bold')

    plt.xlim(0, max(sorted_scores) + 0.08)
    plt.tight_layout()
    plt.savefig("feature_attribution.png", dpi=300, bbox_inches='tight', facecolor='#1e272e')
    plt.close()


if __name__ == "__main__":
    generate_feature_attribution()