# FootCrit-AI: Multimodal Tactical AI Engine

## System Overview
FootCrit-AI is an end-to-end multimodal artificial intelligence architecture designed for elite football tactical analysis. The system bridges the gap between raw, highly-dimensional spatiotemporal tracking data and natural language generation. 

By rejecting high-level abstractions in favor of foundational mathematics, deterministic data pipelines, and graph-theoretic deep learning, this system is capable of mathematically defining pitch geometry, predicting opponent movements, and utilizing a fine-tuned Large Language Model (LLM) to output professional coaching directives.

## Architectural Phases

### 1. Automated ETL Pipeline & Data Lake
* **Ingestion:** Streams 360-degree freeze-frame tracking data and discrete event logs from the StatsBomb Open Data repository.
* **Transformation:** Flattens deeply nested JSON hierarchical graphs into strictly typed, dense 2D columnar arrays using Pandas and PyArrow schemas.
* **Storage:** Serializes the processed data into Snappy-compressed Parquet files, stored in an Amazon S3 data lake utilizing a Hive-style partitioning strategy (`s3://.../competition_id=/season_id=/match_id.parquet`) for optimized I/O throughput.

### 2. Spatial Feature Engineering
Raw coordinate data is transformed into mathematically sound tactical representations prior to modeling:
* **Pitch Control:** Computes a continuous probabilistic surface map integrating ball physics, player momentum, and logistic time-to-intercept functions to quantify spatial ownership.
* **Expected Threat (xT):** Evaluates the offensive value of grid zones by solving a Markov chain recursive formula as a system of linear equations using a direct matrix solver.
* **Network Centrality:** Applies Eigenvector Centrality to passing networks (via NetworkX) to identify the opposition's structural hubs and primary playmakers.

### 3. Spatio-Temporal Graph Neural Network (ST-GNN)
* **Topology:** Models the match state as a directed graph where nodes (players/ball) possess 8-dimensional feature vectors (coordinates, velocity, xT, Pitch Control) and edges represent spatial proximity (Delaunay triangulation).
* **Attention Mechanism:** Implements Graph Attention Network v2 (GATv2) layers via PyTorch Geometric to enable strictly query-dependent dynamic attention, accurately recognizing asymmetrical pressing traps and localized overloads.
* **Temporal Memory:** Aggregates spatial embeddings through a GRU recurrent layer to capture sequential tactical shifts prior to binary classification of defensive vulnerabilities.

### 4. LLM Translation Layer
* **Model:** Google Gemma-2B-it.
* **Methodology:** The dense numerical matrices output by the ST-GNN are serialized into structured JSON prompts. 
* **Fine-Tuning:** The LLM is instruction fine-tuned using Parameter-Efficient Fine-Tuning (PEFT), specifically 4-bit Quantized Low-Rank Adaptation (QLoRA), targeting the query and value attention matrices to translate graph-theoretic geometry into actionable coaching prose.

### 5. Explainability & Interpretability
Black-box decision-making is mitigated through two rigorous explainability frameworks:
* **Subgraph Masking:** Utilizes PyTorch Geometric's `GNNExplainer` to optimize soft masks over node attributes and edges, visually rendering the specific passing lanes and structural overloads that triggered a prediction.
* **Feature Attribution:** Implements Integrated Gradients via the `captum` library, measuring the mathematical area under the gradient curve to prove exactly which spatial variables (e.g., Velocity Magnitude, X-Coordinate depth) drove the network's final activation.

## Repository Structure
* `build_sft_dataset.py` / `multimodal_pipeline.py`: Orchestrates the PEFT/LoRA LLM training and translation layer.
* `comp_&_season_data.py` / `direct_parquet_s3.py`: Handles the extraction, transformation, and S3 Parquet serialization.
* `enterprise_stream.py`: S3 streaming dataloader optimized for PyTorch.
* `st_gnn_model.py` / `inference_stgnn.py`: Core PyTorch definitions for the GATv2 architecture and inference loops.
* `generate_xt.py`: Linear algebra solver for Expected Threat feature engineering.
* `explain_tactics.py`: Generates visual pitch mappings of GATv2 attention masks.
* `feature_importance.py`: Computes axiomatic feature attribution (Integrated Gradients).
* `MODEL_CARD.md`: Documentation of data provenance, physiological biases, and ethical operating boundaries.

## Installation & Execution
```bash
# Initialize the environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Execute the interpretability pipelines
python explain_tactics.py
python feature_importance.py
