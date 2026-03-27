from __future__ import annotations

import gradio as gr
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from src.gui.predictor import ConfigDrivenPredictor

# DEFAULT_WEIGHTS = "src/models/saved_weights/irmas_models/irmas_single_label_mel_cnn/best_val.pt"
DEFAULT_WEIGHTS = "src/models/saved_weights/film_single_label_mel_cnn_ft_irmas/best_val.pt"

# Cache for the predictor to avoid reloading the model on every click
_PREDICTOR_CACHE = {}

def _get_predictor(weights_path: str) -> ConfigDrivenPredictor:
    path_str = str(Path(weights_path).resolve())
    if path_str not in _PREDICTOR_CACHE:
        _PREDICTOR_CACHE[path_str] = ConfigDrivenPredictor(weights_path)
    return _PREDICTOR_CACHE[path_str]

def _feature_to_figure(feature_matrix: np.ndarray, sr: int, hop_length: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    img = librosa.display.specshow(
        feature_matrix,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel", 
        cmap="magma",
        ax=ax,
    )
    ax.set_title("Primary Extracted Feature Representation (First Chunk)")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig

def _run_inference(audio_path: Optional[str], weights_path: str):
    empty_df = pd.DataFrame()
    if not audio_path:
        # Note: Added an extra empty_df here to account for the new summary table output
        return "Please provide an audio clip.", {}, empty_df, None, empty_df, empty_df

    try:
        predictor = _get_predictor(weights_path)
        global_preds, temporal_data, vis_feature = predictor.predict(audio_path)
    except Exception as exc:
        return f"Error: `{exc}`", {}, empty_df, None, empty_df, empty_df

    # 1. Global Overview Data
    label_probs = {label: prob for label, prob in global_preds}
    top_label, top_prob = global_preds[0]
    
    status = (
        f"**Top Prediction:** {top_label} ({top_prob:.1%})\n\n"
        f"**Task Mode:** `{predictor.task_mode}` | **Backbone:** `{predictor.cfg.get('model', {}).get('backbone')}` | "
        f"**Device:** `{predictor.device.type}`\n"
        f"**Audio Sliced Into:** `{len(temporal_data)}` chunks of `{predictor.clip_duration}s`"
    )

    rows = [{"Rank": idx + 1, "Label": l, "Probability": f"{p:.2%}"} for idx, (l, p) in enumerate(global_preds)]
    global_table = pd.DataFrame(rows)
    feature_fig = _feature_to_figure(vis_feature, predictor.sr, predictor.hop)

    # 2. Temporal Data (Raw Probabilities)
    temporal_df = pd.DataFrame(temporal_data)
    
    # Round temporal probabilities for cleaner table reading
    display_df = temporal_df.copy()
    for col in display_df.columns:
        if col != "Time Window":
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")

    # 3. Temporal Summary (Top Class per Chunk)
    # We drop the 'Time Window' column to isolate the probabilities, then find the max
    summary_df = pd.DataFrame({
        "Time Window": temporal_df["Time Window"],
        "Predicted Class": temporal_df.drop(columns=["Time Window"]).idxmax(axis=1)
    })

    return status, label_probs, global_table, feature_fig, summary_df, display_df

def build_interface() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="teal", secondary_hue="slate")
    
    with gr.Blocks(title="Soundtrack Classifier", theme=theme) as demo:
        gr.Markdown("#Soundtrack Classifier")
        
        with gr.Row():
            audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio Input")
            with gr.Column():
                weights_input = gr.Textbox(
                    value=DEFAULT_WEIGHTS,
                    label="Checkpoint Path (.pt)",
                    info="The directory must contain run_config.yaml"
                )
                run_button = gr.Button("Run Prediction", variant="primary")

        status_md = gr.Markdown()

        with gr.Tabs():
            # TAB 1: Global Average
            with gr.Tab("Global Overview"):
                with gr.Row():
                    probs_label = gr.Label(label="Averaged Class Probabilities", num_top_classes=5)
                    preds_table = gr.Dataframe(headers=["Rank", "Label", "Probability"], interactive=False)
            
            # TAB 2: DSP Features
            with gr.Tab("Spectrogram Features"):
                gr.Markdown("The primary digital signal processing representation extracted for the *first* chunk of audio.")
                feature_plot = gr.Plot(label="Feature Visualisation")

            # TAB 3: Segmented Data
            with gr.Tab("Segmented Data"):
                gr.Markdown("### Segment Summary\nThe top predicted class for each time segment.")
                summary_table = gr.Dataframe(interactive=False)
                
                gr.Markdown("### Raw Probabilities\nDetailed probability outputs for every class at every time window.")
                temporal_table = gr.Dataframe(interactive=False)

        run_button.click(
            fn=_run_inference,
            inputs=[audio_input, weights_input],
            outputs=[
                status_md,       # General Status
                probs_label,     # Tab 1: Top probabilities UI
                preds_table,     # Tab 1: Full table
                feature_plot,    # Tab 2: DSP Spectrogram
                summary_table,   # Tab 3: Summary table (NEW)
                temporal_table,  # Tab 3: Chunk DataFrame
            ],
        )

    return demo

if __name__ == "__main__":
    build_interface().launch(share=False)