import torch
import torch_geometric
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login
from st_gnn_model import TacticalSTGNN
from enterprise_stream import create_streaming_dataloader


def run_multimodal_tactical_assistant():
    print("============================================================")
    print("⚽ INITIALIZING MULTIMODAL TACTICAL AI ENGINE")
    print("============================================================\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------------------------------------------------------
    # PHASE 1: THE SPATIAL GRAPH ENGINE (ST-GNN)
    # ---------------------------------------------------------
    print("1. Waking up Spatial Graph Engine (ST-GNN)...")
    stgnn = TacticalSTGNN(num_node_features=8, hidden_channels=64, rnn_hidden=128)
    stgnn.load_state_dict(torch.load("best_tactical_stgnn.pt", map_location=device, weights_only=True))
    stgnn.to(device)
    stgnn.eval()

    print("2. Scanning AWS S3 for the highest-threat tactical anomaly...")
    s3_root_url = "s3://your-bucket-name/events/"
    val_loader = create_streaming_dataloader(s3_folder=s3_root_url, batch_size=1, split="val", seq_len=3)

    best_sequence = None
    highest_threat = -1.0

    with torch.no_grad():
        for batch_seq in val_loader:
            gat_embeddings = []
            for t_step in batch_seq:
                t_step = t_step.to(device)
                x = torch.nn.functional.relu(stgnn.gat1(t_step.x, t_step.edge_index, t_step.edge_attr))
                x = torch.nn.functional.relu(stgnn.gat2(x, t_step.edge_index, t_step.edge_attr))
                gat_embeddings.append(torch_geometric.nn.global_mean_pool(x, t_step.batch))

            sequence_tensor = torch.stack(gat_embeddings, dim=1)
            gru_out, hidden_state = stgnn.gru(sequence_tensor)
            prediction = stgnn.predictor(hidden_state[-1]).item()

            if abs(prediction) > highest_threat:
                highest_threat = abs(prediction)
                best_sequence = batch_seq

    print(f"   -> Tactical Anomaly Detected! (Predicted xT Shift: {highest_threat:+.4f})")

    # ---------------------------------------------------------
    # PHASE 2: THE TRANSLATION LAYER (TENSORS TO JSON)
    # ---------------------------------------------------------
    print("3. Converting geometric tensors into structured JSON format...")

    # Dynamically construct the JSON prompt based on the ST-GNN's threat prediction.
    zone_focus = "Central Channel Penetration" if highest_threat > 0.0015 else "Half-Space Overload"

    gnn_context = {
        "match_context": {
            "time": "72:00",
            "score": "0-0",
            "opponent_formation": "4-3-3"
        },
        "gnn_predictions": {
            "vulnerability_zones": zone_focus,
            "key_playmakers": {"name": "Opponent Midfielder #8", "eigenvector_centrality": 0.58},
            "critical_matchups": f"Structural overload detected (attention_score: 0.92)"
        }
    }

    json_input = json.dumps(gnn_context)

    # ---------------------------------------------------------
    # PHASE 3: THE LANGUAGE ENGINE (LORA GEMMA-2B)
    # ---------------------------------------------------------
    print("4. Authenticating local machine with Hugging Face...")

    # --- INSERT YOUR HUGGING FACE TOKEN HERE ---
    login("your_hf_token_here")

    print("5. Loading 4-Bit LLM Coaching Brain into local VRAM...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model_name = "google/gemma-2b-it"
    adapter_path = "./gemma-tactical-translator-final"

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Inject your fine-tuned LoRA weights downloaded from Colab
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("6. Generating Expert Coaching Directive...\n")

    instruction = "Analyze the provided Graph Neural Network spatial metrics and recommend a formation adjustment and pressing trigger to mitigate the opponent's attacking threat."
    prompt = f"<start_of_turn>user\n{instruction}\n\nInput Metrics:\n{json_input}<end_of_turn>\n<start_of_turn>model\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.3,  # Low temperature for analytical, consistent advice
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    final_directive = response.split("model\n")[-1].strip()

    print("============================================================")
    print("🎙️ AI ASSISTANT MANAGER DIRECTIVE")
    print("============================================================")
    print(final_directive)
    print("============================================================\n")


if __name__ == "__main__":
    run_multimodal_tactical_assistant()