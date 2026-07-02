import json
import random


def generate_synthetic_tactical_dataset(num_samples=500, output_file="tactical_sft_dataset.jsonl"):
    """
    Generates a Supervised Fine-Tuning (SFT) dataset to train an LLM
    to translate ST-GNN spatial tensors into natural language coaching directives.
    """
    print(f"1. Initializing Multimodal Translation Layer...")
    print(f"2. Generating {num_samples} synthetic ST-GNN output scenarios...")

    dataset = []

    zones = ["Zone 14", "Right Half-Space", "Left Flank", "Central Channel Penetration", "Defensive Third"]
    playmakers = ["Player A", "Player B", "Player C", "Player D"]
    formations = ["4-3-3", "4-2-3-1", "3-5-2", "4-4-2 Low Block"]

    for i in range(num_samples):
 
        vulnerability = random.choice(zones)
        attention_score = round(random.uniform(0.75, 0.98), 2)
        eigen_centrality = round(random.uniform(0.30, 0.65), 2)
        target_playmaker = random.choice(playmakers)
        opp_formation = random.choice(formations)

  
        gnn_input_context = {
            "match_context": {
                "time": f"{random.randint(10, 85)}:00",
                "score": f"{random.randint(0, 2)}-{random.randint(0, 2)}",
                "opponent_formation": opp_formation
            },
            "gnn_predictions": {
                "vulnerability_zones": vulnerability,
                "key_playmakers": {"name": target_playmaker, "eigenvector_centrality": eigen_centrality},
                "critical_matchups": f"{target_playmaker} overloads defensive structure (attention_score: {attention_score})"
            }
        }


        if "Flank" in vulnerability or "Half-Space" in vulnerability:
            adjustment = "Shift to an asymmetric defensive block."
            trigger = f"Drop the winger into a deeper position to eliminate the spatial overload detected in the {vulnerability}."
        else:
            adjustment = "Compress the vertical spacing between the midfield and defensive lines."
            trigger = f"Instruct the defensive midfielder to man-mark {target_playmaker} to disrupt their passing hub."

        expert_output = (
            f"The opponent's primary playmaker ({target_playmaker}) possesses a high eigenvector centrality of {eigen_centrality}, "
            f"dictating play effectively. Concurrently, the ST-GNN predicts a severe vulnerability in the {vulnerability} "
            f"with an {attention_score} attention weight indicating spatial isolation. "
            f"Recommendation: {adjustment} {trigger}"
        )

   
        sft_row = {
            "instruction": "Analyze the provided Graph Neural Network spatial metrics and recommend a formation adjustment and pressing trigger to mitigate the opponent's attacking threat.",
            "input": json.dumps(gnn_input_context),
            "output": expert_output
        }

        dataset.append(sft_row)


    
    with open(output_file, 'w') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')

    print(f"\nSUCCESS: Dataset saved to '{output_file}'.")
    print("Ready for PEFT/LoRA fine-tuning using the HuggingFace Transformers library.")


if __name__ == "__main__":
    generate_synthetic_tactical_dataset()
