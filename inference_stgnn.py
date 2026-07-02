import torch
import torch_geometric
from st_gnn_model import TacticalSTGNN
from enterprise_stream import create_streaming_dataloader


def run_temporal_inference():
    print("1. Initializing Spatio-Temporal AI Blueprint...")
    model = TacticalSTGNN(num_node_features=8, hidden_channels=64, rnn_hidden=128)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("2. Loading Temporal Memory Weights (ST-GNN)...")
    try:
        model.load_state_dict(torch.load("best_tactical_stgnn.pt", map_location=device, weights_only=True))
        print("   -> Weights loaded successfully!")
    except FileNotFoundError:
        print("   -> [WARNING]: 'best_tactical_stgnn.pt' not found. Using untrained weights.")

    model.to(device)
    model.eval()

    print("\n3. Fetching unseen chronological sequences from AWS S3...")
    s3_root_url = "s3://your-bucket-name/events/"
    val_loader = create_streaming_dataloader(s3_folder=s3_root_url, batch_size=1, split="val", seq_len=3)

    print("\n4. Running AI Temporal Tactical Assessment (TOP-5 SCOUTING MODE)...")
    print("   -> Scanning the entire validation set in the background. Please wait...\n")

    scout_reports = []

    with torch.no_grad():
        for i, batch_seq in enumerate(val_loader):
            gat_embeddings = []

            # Process Spatial Frames
            for t_step in batch_seq:
                t_step = t_step.to(device)
                x = torch.nn.functional.relu(model.gat1(t_step.x, t_step.edge_index, t_step.edge_attr))
                x = torch.nn.functional.relu(model.gat2(x, t_step.edge_index, t_step.edge_attr))
                pooled_x = torch_geometric.nn.global_mean_pool(x, t_step.batch)
                gat_embeddings.append(pooled_x)

            # Temporal Sequencing and Prediction
            sequence_tensor = torch.stack(gat_embeddings, dim=1)
            gru_out, hidden_state = model.gru(sequence_tensor)
            prediction = model.predictor(hidden_state[-1])

            true_xt = batch_seq[-1].y.item()
            predicted_xt = prediction.item()

            # Save every sequence to our scouting list
            scout_reports.append({
                "id": i + 1,
                "true_xt": true_xt,
                "predicted_xt": predicted_xt
            })

    # Sort the reports by the absolute value of the predicted threat (Highest to Lowest)
    scout_reports.sort(key=lambda x: abs(x["predicted_xt"]), reverse=True)

    print("=" * 60)
    print(f"📊 AI SCOUTING REPORT: TOP 5 HIGHEST THREAT SEQUENCES")
    print("=" * 60)
    print(f"Total sequences analyzed: {len(scout_reports)}\n")

    # Print the Top 5 most extreme predictions
    for rank, report in enumerate(scout_reports[:5]):
        print(f"RANK #{rank + 1} | Chronological Sequence #{report['id']}")
        print(f"Final Frame True xT:      {report['true_xt']:+.6f}")
        print(f"AI Predicted Future xT:   {report['predicted_xt']:+.6f}")

        # Dynamic Assessment based on the new scale
        if report['predicted_xt'] > 0.001:
            print("🧠 ST-GNN Assessment: POSITIVE FORWARD MOMENTUM")
        elif report['predicted_xt'] < -0.001:
            print("🧠 ST-GNN Assessment: NEGATIVE MOMENTUM / TURNOVER RISK")
        else:
            print("🧠 ST-GNN Assessment: MINOR POSSESSION SHIFT")
        print("-" * 60)


if __name__ == "__main__":
    run_temporal_inference()