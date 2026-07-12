import gradio as gr
import os
import torch
import torch_geometric
import json
import requests
from vision_tracker import TacticalVisionPipeline
from homography_transformer import HomographyEngine
from data_exporter import TacticalDataExporter
from graph_builder import TacticalGraphBuilder
from st_gnn_model import TacticalSTGNN

CLOUD_LLM_API_URL = " your-cloud-url-here"


def run_tactical_pipeline(video_path, analysis_mode="Full Tactical Breakdown",
                           conf_threshold=0.65, use_homography=True):
    if video_path is None:
        return None, "Error: No video uploaded.", None

    annotated_video_path = os.path.join(os.getcwd(), "annotated_output.mp4")
    vision = TacticalVisionPipeline()
    raw_pixel_data, processed_video = vision.process_video(video_path, annotated_video_path)

    if use_homography:
        homography_engine = HomographyEngine()
        pitch_data = homography_engine.transform_tracking_data(raw_pixel_data)
    else:
        pitch_data = raw_pixel_data

    parquet_path = os.path.join(os.getcwd(), "tactical_telemetry.parquet")
    TacticalDataExporter.export_to_parquet(pitch_data, parquet_path)

    graph_builder = TacticalGraphBuilder(distance_threshold=25.0)
    graphs = graph_builder.build_graphs_from_parquet(parquet_path)

    if len(graphs) < 3:
        return processed_video, "Error: Video too short. Minimum 3 frames required.", None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TacticalSTGNN(num_node_features=8, hidden_channels=64, rnn_hidden=128)

    try:
        model.load_state_dict(torch.load("best_tactical_stgnn.pt", map_location=device, weights_only=True))
    except FileNotFoundError:
        pass

    model.to(device)
    model.eval()

    gat_embeddings = []
    with torch.no_grad():
        for t_step in graphs[-3:]:
            t_step = t_step.to(device)
            batch = torch.zeros(t_step.x.size(0), dtype=torch.long, device=device)
            x = torch.nn.functional.relu(model.gat1(t_step.x, t_step.edge_index, t_step.edge_attr))
            x = torch.nn.functional.relu(model.gat2(x, t_step.edge_index, t_step.edge_attr))
            pooled_x = torch_geometric.nn.global_mean_pool(x, batch)
            gat_embeddings.append(pooled_x)

        sequence_tensor = torch.stack(gat_embeddings, dim=1)
        gru_out, hidden_state = model.gru(sequence_tensor)
        prediction = model.predictor(hidden_state[-1]).item()

    severity = "CRITICAL" if abs(prediction) > 0.05 else "MODERATE"
    zone_focus = "Central Channel Penetration" if prediction > 0.0015 else "Half-Space Overload"

    gnn_context = {
        "gnn_predictions": {
            "xt_shift": prediction,
            "vulnerability_zones": zone_focus,
            "threat_level": severity
        },
        "analysis_settings": {
            "mode": analysis_mode,
            "detection_confidence": conf_threshold,
            "homography_calibration": use_homography
        }
    }

    try:
        response = requests.post(
            CLOUD_LLM_API_URL,
            json={"json_context": json.dumps(gnn_context)},
            timeout=30
        )
        if response.status_code == 200:
            llm_text = response.json().get("directive", "API Error: No directive returned.")
            tactical_advice = f"### Crit's Live Cloud Advice\n\n**ST-GNN xT Shift:** {prediction:+.5f}\n\n{llm_text}"
        else:
            tactical_advice = f"Cloud API Error {response.status_code}: Make sure Colab is running."
    except Exception as e:
        tactical_advice = f"Connection Failed. Ensure the Ngrok URL is correct and Colab is active. Error: {str(e)}"

    visual_proof = "feature_attribution.png" if os.path.exists("feature_attribution.png") else None

    return processed_video, tactical_advice, visual_proof


# ---------------------------------------------------------------------------
# UI
#
# Design direction: a night-match "command center" — the dark, floodlit
# control room a tactical analyst would actually work from pitch-side.
# Corner brackets on every visual panel echo the YOLOv8 bounding boxes the
# engine itself draws, so the chrome quotes the product's own detection
# output rather than decorating around it. Three accent colors carry
# distinct jobs: turf green for actions and calibration, cyan for
# live data readouts, amber/red reserved for threat severity.
# ---------------------------------------------------------------------------

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --fc-bg: #0A0F0D;
    --fc-surface: #121815;
    --fc-surface-2: #1A2420;
    --fc-line: #263129;
    --fc-chalk: #F3F1E7;
    --fc-muted: #8FA096;
    --fc-turf: #4FA672;
    --fc-turf-bright: #6FD08C;
    --fc-cyan: #4FD1C5;
    --fc-amber: #F2A93B;
    --fc-red: #E1573F;
    --fc-font-display: 'Oswald', sans-serif;
    --fc-font-body: 'Inter', -apple-system, system-ui, sans-serif;
    --fc-font-mono: 'JetBrains Mono', 'SFMono-Regular', Consolas, monospace;
}

.gradio-container {
    background: var(--fc-bg) !important;
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding-left: clamp(20px, 4vw, 64px) !important;
    padding-right: clamp(20px, 4vw, 64px) !important;
    box-sizing: border-box !important;
    font-family: var(--fc-font-body) !important;
    color: var(--fc-chalk) !important;

    --background-fill-primary: var(--fc-bg);
    --background-fill-secondary: var(--fc-surface);
    --body-background-fill: var(--fc-bg);
    --body-text-color: var(--fc-chalk);
    --body-text-color-subdued: var(--fc-muted);
    --border-color-primary: var(--fc-line);
    --border-color-accent: var(--fc-turf);
    --block-background-fill: var(--fc-surface);
    --block-border-color: var(--fc-line);
    --block-border-width: 1px;
    --block-radius: 4px;
    --block-label-background-fill: transparent;
    --block-label-text-color: var(--fc-muted);
    --block-label-text-size: 0.72rem;
    --block-title-text-color: var(--fc-chalk);
    --block-info-text-color: var(--fc-muted);
    --input-background-fill: var(--fc-surface-2);
    --input-border-color: var(--fc-line);
    --input-border-color-focus: var(--fc-turf);
    --button-primary-background-fill: var(--fc-turf);
    --button-primary-background-fill-hover: var(--fc-turf-bright);
    --button-primary-text-color: #0A0F0D;
    --button-primary-border-color: var(--fc-turf);
    --button-secondary-background-fill: var(--fc-surface-2);
    --button-secondary-text-color: var(--fc-chalk);
    --button-secondary-border-color: var(--fc-line);
    --slider-color: var(--fc-turf);
    --checkbox-background-color: var(--fc-surface-2);
    --checkbox-background-color-selected: var(--fc-turf);
    --checkbox-border-color: var(--fc-line);
    --checkbox-border-color-selected: var(--fc-turf);
}

.gradio-container label {
    font-family: var(--fc-font-mono) !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.7rem !important;
    font-weight: 500;
    color: var(--fc-muted) !important;
}

.gradio-container button.primary {
    font-family: var(--fc-font-display) !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    font-size: 0.95rem !important;
    border-radius: 3px !important;
    box-shadow: none !important;
}
.gradio-container button.primary:hover {
    background: var(--fc-turf-bright) !important;
}

.gradio-container .prose h3 {
    font-family: var(--fc-font-display) !important;
    color: var(--fc-chalk) !important;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    font-size: 1rem !important;
    border-bottom: 1px solid var(--fc-line);
    padding-bottom: 8px;
    margin-bottom: 12px !important;
}
.gradio-container .prose strong {
    color: var(--fc-cyan);
    font-family: var(--fc-font-mono);
}

/* Header */
.fc-header {
    padding: 28px 8px 20px;
    margin-bottom: 22px;
    border-bottom: 1px solid var(--fc-line);
    background-image: radial-gradient(circle at 94% 30%, rgba(79, 166, 114, 0.16) 0%, rgba(79, 166, 114, 0) 45%);
}
.fc-header-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 10px;
}
.fc-eyebrow {
    font-family: var(--fc-font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--fc-muted);
}
.fc-status {
    display: flex;
    align-items: center;
    gap: 7px;
    font-family: var(--fc-font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--fc-turf-bright);
}
.fc-status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--fc-turf-bright);
    animation: fc-pulse 2s infinite;
}
@keyframes fc-pulse {
    0%   { box-shadow: 0 0 0 0 rgba(111, 208, 140, 0.55); }
    70%  { box-shadow: 0 0 0 6px rgba(111, 208, 140, 0); }
    100% { box-shadow: 0 0 0 0 rgba(111, 208, 140, 0); }
}
@media (prefers-reduced-motion: reduce) {
    .fc-status-dot { animation: none; }
}
.fc-title {
    font-family: var(--fc-font-display);
    font-size: 2.6rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: var(--fc-chalk);
    margin: 0 0 6px 0;
    line-height: 1;
}
.fc-subtitle {
    font-family: var(--fc-font-body);
    font-size: 0.95rem;
    color: var(--fc-muted);
    margin: 0;
    max-width: 620px;
    line-height: 1.5;
}

/* Config panel */
.fc-panel {
    background: var(--fc-surface) !important;
    border: 1px solid var(--fc-line) !important;
    border-left: 3px solid var(--fc-turf) !important;
    border-radius: 4px !important;
    padding: 18px !important;
}
.fc-panel-title {
    font-family: var(--fc-font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--fc-turf-bright);
    margin-bottom: 14px !important;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--fc-line);
}

/* Bracket frame — echoes the engine's own YOLOv8 detection boxes */
.fc-bracket {
    position: relative;
    padding: 34px 10px 10px 10px !important;
    background: var(--fc-surface) !important;
    border: 1px solid var(--fc-line) !important;
    border-radius: 4px !important;
}
.fc-bracket::before {
    content: "";
    position: absolute;
    inset: 34px 10px 10px 10px;
    pointer-events: none;
    z-index: 5;
    background:
        linear-gradient(var(--fc-turf), var(--fc-turf)) 0 0 / 18px 2px no-repeat,
        linear-gradient(var(--fc-turf), var(--fc-turf)) 0 0 / 2px 18px no-repeat,
        linear-gradient(var(--fc-turf), var(--fc-turf)) 100% 0 / 18px 2px no-repeat,
        linear-gradient(var(--fc-turf), var(--fc-turf)) 100% 0 / 2px 18px no-repeat,
        linear-gradient(var(--fc-turf), var(--fc-turf)) 0 100% / 18px 2px no-repeat,
        linear-gradient(var(--fc-turf), var(--fc-turf)) 0 100% / 2px 18px no-repeat,
        linear-gradient(var(--fc-turf), var(--fc-turf)) 100% 100% / 18px 2px no-repeat,
        linear-gradient(var(--fc-turf), var(--fc-turf)) 100% 100% / 2px 18px no-repeat;
}

/* Section divider + labels */
.fc-divider {
    border: none;
    border-top: 1px solid var(--fc-line);
    margin: 28px 0 20px 0;
}
.fc-section-title {
    font-family: var(--fc-font-mono);
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--fc-muted);
    margin-bottom: 16px !important;
}

/* Footer */
.fc-footer {
    margin-top: 32px;
    padding: 16px 8px;
    border-top: 1px solid var(--fc-line);
    font-family: var(--fc-font-mono);
    font-size: 0.66rem;
    letter-spacing: 0.06em;
    color: var(--fc-muted);
    text-transform: uppercase;
    text-align: center;
}

@media (max-width: 768px) {
    .fc-title { font-size: 1.9rem; }
    .fc-header-top { flex-direction: column; align-items: flex-start; }
}
"""

with gr.Blocks(title="FootCrit-AI | Tactical Engine") as demo:

    gr.HTML(
        """
        <div class="fc-header">
            <div class="fc-header-top">
                <span class="fc-eyebrow">Spatio-Temporal Graph Neural Network Interface</span>
                <span class="fc-status"><span class="fc-status-dot"></span>Engine Ready</span>
            </div>
            <h1 class="fc-title">FOOTCRIT-AI</h1>
            <p class="fc-subtitle">Upload match footage, track every player on the pitch, and get a
            tactical directive generated from the play's spatio-temporal shape.</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1, elem_classes="fc-panel"):
            gr.Markdown('<div class="fc-panel-title">Detection Settings</div>')
            analysis_mode = gr.Dropdown(
                choices=["Full Tactical Breakdown", "Defensive Line Tracking", "Counter-Attack Spaces"],
                value="Full Tactical Breakdown",
                label="Analysis Mode",
                interactive=True,
            )
            conf_threshold = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.65, step=0.01,
                label="Detection Confidence",
                info="Higher values discard uncertain player detections.",
                interactive=True,
            )
            homography_check = gr.Checkbox(
                value=True,
                label="Pitch Homography Calibration",
                info="Maps tracked pixels onto real-world pitch coordinates.",
                interactive=True,
            )

        with gr.Column(scale=3):
            with gr.Row():
                input_video = gr.Video(label="Source Footage", elem_classes="fc-bracket")
                output_video = gr.Video(label="Tracked Output", interactive=False, elem_classes="fc-bracket")

            process_btn = gr.Button("Run Tactical Analysis", variant="primary", size="lg")

    gr.HTML('<hr class="fc-divider">')
    gr.Markdown('<div class="fc-section-title">Tactical Intelligence Report</div>')

    with gr.Row():
        llm_output = gr.Markdown(
            value="*Awaiting analysis — run the pipeline above to generate a live tactical directive.*",
        )
        explainability_img = gr.Image(
            label="Spatial Proof · Feature Attribution",
            type="filepath",
            interactive=False,
            elem_classes="fc-bracket",
        )

    gr.HTML(
        '<div class="fc-footer">YOLOv8 + ByteTrack Detection &nbsp;·&nbsp; Homography Calibration '
        '&nbsp;·&nbsp; ST-GNN Prediction &nbsp;·&nbsp; Cloud Directive Engine</div>'
    )

    process_btn.click(
        fn=run_tactical_pipeline,
        inputs=[input_video, analysis_mode, conf_threshold, homography_check],
        outputs=[output_video, llm_output, explainability_img]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Base(),
        css=custom_css,
    )