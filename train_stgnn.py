import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric

from enterprise_stream import create_streaming_dataloader
from st_gnn_model import TacticalSTGNN


def threat_weighted_mse(predictions, targets, alpha=5.0):
    """
    Penalizes the AI more heavily when it gets high-threat situations wrong.
    alpha controls how aggressive the penalty is.
    """
  
    base_error = (predictions - targets) ** 2


    weight = 1.0 + (alpha * torch.abs(targets))


    weighted_loss = base_error * weight
    return torch.mean(weighted_loss)


def train_stgnn(model, train_loader, val_loader, epochs=10, lr=0.0001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')

    for epoch in range(epochs):

        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_seq in train_loader:
           
            optimizer.zero_grad()

            gat_embeddings = []

            for t_step in batch_seq:
                t_step = t_step.to(device)
                x = torch.nn.functional.relu(model.gat1(t_step.x, t_step.edge_index, t_step.edge_attr))
                x = torch.nn.functional.relu(model.gat2(x, t_step.edge_index, t_step.edge_attr))
                pooled_x = torch_geometric.nn.global_mean_pool(x, t_step.batch)
                gat_embeddings.append(pooled_x)

      
            sequence_tensor = torch.stack(gat_embeddings, dim=1)

            gru_out, hidden_state = model.gru(sequence_tensor)

            predictions = model.predictor(hidden_state[-1])

            target_y = batch_seq[-1].y.to(device)

            pred_scaled = predictions.squeeze() * 100
            targ_scaled = target_y.squeeze() * 100
            loss = threat_weighted_mse(pred_scaled, targ_scaled, alpha=5.0)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_seq in val_loader:
                gat_embeddings = []
                for t_step in batch_seq:
                    t_step = t_step.to(device)
                    x = torch.nn.functional.relu(model.gat1(t_step.x, t_step.edge_index, t_step.edge_attr))
                    x = torch.nn.functional.relu(model.gat2(x, t_step.edge_index, t_step.edge_attr))
                    pooled_x = torch_geometric.nn.global_mean_pool(x, t_step.batch)
                    gat_embeddings.append(pooled_x)

                sequence_tensor = torch.stack(gat_embeddings, dim=1)
                gru_out, hidden_state = model.gru(sequence_tensor)
                predictions = model.predictor(hidden_state[-1])
                target_y = batch_seq[-1].y.to(device)

                pred_scaled = predictions.squeeze() * 100
                targ_scaled = target_y.squeeze() * 100
                loss = threat_weighted_mse(pred_scaled, targ_scaled, alpha=5.0)

                val_loss += loss.item()
                val_batches += 1

        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0

        print(f"Epoch {epoch + 1:03d} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print("   -> Validation loss improved! Saving ST-GNN weights...")
            torch.save(model.state_dict(), "best_tactical_stgnn.pt")

    print("\nTraining Complete! Best model saved to best_tactical_stgnn.pt")


if __name__ == "__main__":
    s3_root_url = "s3://sports-ai-datalake-emon/events/"
    train_loader = create_streaming_dataloader(s3_folder=s3_root_url, batch_size=32, split="train")
    val_loader = create_streaming_dataloader(s3_folder=s3_root_url, batch_size=32, split="val")


    model = TacticalSTGNN(num_node_features=8, hidden_channels=64, rnn_hidden=128)
    train_stgnn(model, train_loader, val_loader, epochs=20)
