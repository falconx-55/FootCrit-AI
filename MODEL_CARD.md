# Model Card: Multimodal Tactical AI Engine (ST-GNN + LLM)

## Model Details
* **Architecture:** * **Spatial Engine:** Spatio-Temporal Graph Neural Network (ST-GNN) featuring Masked Attention (GATv2) and GRU Temporal Memory.
  * **Translation Layer:** Google Gemma-2B-it (Large Language Model).
  * **Integration:** LoRA (Low-Rank Adaptation) adapters injected into the LLM via 4-bit Quantization.
* **Objective:** To ingest raw Spatio-Temporal football tracking data, predict structural defensive vulnerabilities based on Expected Threat (xT) shifts, and generate explainable, natural language coaching directives.
* **Frameworks:** PyTorch, PyTorch Geometric (PyG), Hugging Face Transformers, PEFT, TRL.

## Intended Use & Ethical Boundaries
* **Primary Use Case:** This system is engineered exclusively as an **augmentation tool** for human football analysts and coaching staff. It processes multi-agent relational topologies to identify spatial overloads and key passing network centralities.
* **Out-of-Scope Use:** This model is **not an autonomous decision-maker**. It does not possess the capability to account for unquantifiable human elements crucial to match management.
* **Limitations:** The model operates strictly under a framework of spatial rationality. It cannot compute or evaluate:
  * Biological fatigue or player injury status.
  * Psychological momentum shifts or team morale.
  * Extraneous physical conditions (e.g., pitch degradation, weather anomalies).

## Training Data & Provenance
* **Data Source:** StatsBomb Open Data Repository.
* **Modality:** 360-degree freeze-frame spatial coordinate data and discrete event logs.
* **Preprocessing:** Raw JSON nested dictionaries were flattened into dense columnar arrays via Pandas and converted to PyArrow schemas for Parquet storage on AWS S3.
* **Spatial Feature Engineering:** Features encompass raw velocity coordinates, Pitch Control integration mapping, and Expected Threat (xT) values computed via linear algebra direct solvers.

## Bias & Generalization Profiling
* **Physiological Bias:** The ST-GNN was trained predominantly on top-tier European men's tracking data. Consequently, the spatial valuations (e.g., velocity maximums, arrival probabilities) are strictly biased toward elite-level male physiological profiles.
* **Degradation of Accuracy:** Application of this model to domains with differing physiological constraints—such as women's professional leagues, lower-tier divisions, or youth academies—will result in degraded predictive accuracy, as the assumed spatial constraints and sprint speeds will misalign with the realities of those leagues.
* **Sensor Variance:** The kinematics calculations rely on a 2D projection of 3D space derived from broadcast tracking data (25 fps). This carries a significantly wider margin of error for acceleration and velocity metrics compared to professional optical stadium arrays.

## Explainability Standards
* **Algorithm:** Interpretability is achieved via the `GNNExplainer` module (`torch_geometric.explain`).
* **Mechanism:** The explainer optimizes a soft mask over the target graph to isolate the specific `edge_mask` (passing networks/isolations) and `node_mask` (player influence) that maximized mutual information with the final tactical prediction.
* **Verification:** Users can visually verify AI logic by rendering the masked sub-graph to guarantee the LLM's natural language output is mathematically grounded in the tracked spatial geometry.
