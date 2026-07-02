import torch
import torch_geometric
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from st_gnn_model import TacticalSTGNN
from enterprise_stream import create_streaming_dataloader


def visualize_top_sequence():
    print("1. Initializing Scout Visualizer...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TacticalSTGNN(num_node_features=8, hidden_channels=64, rnn_hidden=128)
    try:
        model.load_state_dict(torch.load("best_tactical_stgnn.pt", map_location=device, weights_only=True))
    except FileNotFoundError:
        print("[ERROR]: best_tactical_stgnn.pt not found. Run train_stgnn.py first.")
        return

    model.to(device)
    model.eval()

    print("2. Scanning AWS S3 for the highest-threat sequence...")
    s3_root_url = "s3://your_bucket_name/events/"
    val_loader = create_streaming_dataloader(s3_folder=s3_root_url, batch_size=1, split="val", seq_len=3)

    best_sequence = None
    highest_threat_magnitude = -1.0
    best_prediction = 0.0

    with torch.no_grad():
        for batch_seq in val_loader:
            gat_embeddings = []
            for t_step in batch_seq:
                t_step = t_step.to(device)
                x = torch.nn.functional.relu(model.gat1(t_step.x, t_step.edge_index, t_step.edge_attr))
                x = torch.nn.functional.relu(model.gat2(x, t_step.edge_index, t_step.edge_attr))
                gat_embeddings.append(torch_geometric.nn.global_mean_pool(x, t_step.batch))

            sequence_tensor = torch.stack(gat_embeddings, dim=1)
            gru_out, hidden_state = model.gru(sequence_tensor)
            prediction = model.predictor(hidden_state[-1]).item()

            if abs(prediction) > highest_threat_magnitude:
                highest_threat_magnitude = abs(prediction)
                best_sequence = batch_seq
                best_prediction = prediction

    if best_sequence is None:
        print("No sequences found.")
        return

    print(f"\n3. Top Sequence Found! AI Predicted xT: {best_prediction:+.5f}")
    print("   -> Rendering Tactical Pitch...")

    final_frame = best_sequence[-1].cpu()

    # Feature map: [0:x, 1:y, 2:is_team, 3:pos, 4:is_carrier, 5:press, 6:dist_to_goal, 7:angle]
    x_coords = final_frame.x[:, 0].numpy() * 120.0
    y_coords = final_frame.x[:, 1].numpy() * 80.0
    teammates = final_frame.x[:, 2].numpy()
    is_carrier = final_frame.x[:, 4].numpy()
    edges = final_frame.edge_index.numpy()
    true_xt = final_frame.y.item()


    pitch = Pitch(pitch_type='statsbomb', pitch_color='#1a2421', line_color='#c7d5cc')
    fig, ax = pitch.draw(figsize=(10, 7))
    fig.set_facecolor('#1a2421')


    for i in range(edges.shape[1]):
        src, dst = edges[0, i], edges[1, i]
        pitch.lines(x_coords[src], y_coords[src], x_coords[dst], y_coords[dst],
                    ax=ax, color='white', alpha=0.3, zorder=1)


    opp_mask = (teammates == 0.0)
    pitch.scatter(x_coords[opp_mask], y_coords[opp_mask], ax=ax, c='#1e90ff',
                  s=150, edgecolors='black', linewidth=2, zorder=2, label="Defending Team")

    team_mask = (teammates == 1.0) & (is_carrier == 0.0)
    pitch.scatter(x_coords[team_mask], y_coords[team_mask], ax=ax, c='#ea6969',
                  s=150, edgecolors='black', linewidth=2, zorder=2, label="Attacking Team")


    carrier_mask = (is_carrier == 1.0)
    pitch.scatter(x_coords[carrier_mask], y_coords[carrier_mask], ax=ax, c='#ffd700',
                  s=250, edgecolors='black', linewidth=2, zorder=3, label="Ball Carrier", marker='*')


    plt.title(f"ST-GNN Memory Frame | Predicted xT: {best_prediction:+.4f} | True xT: {true_xt:+.4f}",
              color='white', fontsize=14)
    legend = ax.legend(loc='upper left', facecolor='#1a2421', edgecolor='white')
    for text in legend.get_texts(): text.set_color("white")

    plt.show()


if __name__ == "__main__":
    visualize_top_sequence()
