from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple

from src.utils.system_utils import load_yaml, get_repo_root, resolve_path
from src.models.builder import build_model
from src.preprocessing.feature_modes import (
    normalize_feature_mode,
    feature_mode_to_features,
    feature_mode_to_in_channels,
    align_and_stack_feature_tensors,
)
from src.preprocessing.features import (
    compute_stereo_logmel_db,
    compute_stereo_cqt_db,
    compute_stereo_mfcc,
    compute_stereo_chroma,
    compute_stft_params,
)
from src.preprocessing.audio_io import (
    load_audio_as_stereo_and_resample,
    preprocess_loudness,
)


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ConfigDrivenPredictor:
    """
    Dynamically loads a model and its required preprocessing pipeline 
    based on the run_config.yaml saved alongside the checkpoint.
    """
    def __init__(self, checkpoint_path: str | Path):
        self.root = get_repo_root()
        self.checkpoint_path = resolve_path(checkpoint_path, self.root)
        self.device = _select_device()
        
        # Resolve config
        run_dir = self.checkpoint_path.parent
        config_path = run_dir / "run_config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Cannot find run_config.yaml in {run_dir}")
            
        self.cfg = load_yaml(config_path)
        self.task_mode = self.cfg.get("task_mode", "single_label")
        self.classes = self.cfg.get("classes", [])
        self.audio_params = self.cfg.get("audio_params", {})
        self.feature_mode = normalize_feature_mode(self.cfg.get("feature_mode", "mel"))
        self.feature_names = feature_mode_to_features(self.feature_mode)
        
        # Load Model
        in_channels = feature_mode_to_in_channels(self.feature_mode)
        model_cfg = self.cfg.get("model", {})
        backbone = model_cfg.get("backbone", "cnn")
        
        self.model = build_model(backbone, in_channels, len(self.classes), model_cfg)
        
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt.get("model_state", ckpt), strict=True)
        self.model.to(self.device)
        self.model.eval()

        # DSP Setup matching SlidingWindowTestDataset
        self.sr = int(self.audio_params.get("sr", 44100))
        self.clip_duration = float(self.audio_params.get("duration", 3.0))
        self.chunk_samples = int(self.clip_duration * self.sr)
        
        self.n_fft, self.hop, self.win_length = compute_stft_params(
            self.sr, 
            float(self.audio_params.get("win_ms", 30.0)), 
            float(self.audio_params.get("hop_ms", 10.0))
        )
        
        # Extended DSP parameters exactly as defined in run_eval.py
        self.window = str(self.audio_params.get("window", "hann"))
        self.fmin = float(self.audio_params.get("fmin", 20.0))
        self.fmax = float(self.audio_params.get("fmax", self.sr / 2))
        self.n_mels = int(self.audio_params.get("n_mels", 128))
        self.n_bins = int(self.audio_params.get("n_bins", 120))
        self.bins_per_octave = int(self.audio_params.get("bins_per_octave", 12))
        self.n_mfcc = int(self.audio_params.get("n_mfcc", 13))
        self.n_chroma = int(self.audio_params.get("n_chroma", 12))
        self.feature_norm = str(self.audio_params.get("feature_norm", "none")).lower()

    def _apply_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Applies normalisation to match training distribution."""
        if self.feature_norm == "min_max":
            x_min, x_max = x.min(), x.max()
            if x_max - x_min > 1e-8: 
                return (x - x_min) / (x_max - x_min)
        elif self.feature_norm == "standard":
            return (x - x.mean()) / (x.std() + 1e-8)
        return x

    def _extract_single_feature(self, stereo: np.ndarray, feature_name: str) -> torch.Tensor:
        """Extracts a specific feature type using identical parameters to evaluation."""
        if feature_name == "mel":
            feat = torch.from_numpy(compute_stereo_logmel_db(
                stereo, self.sr, n_fft=self.n_fft, hop=self.hop, 
                win_length=self.win_length, n_mels=self.n_mels, 
                fmin=self.fmin, fmax=self.fmax, window=self.window
            )).float()
        elif feature_name == "cqt":
            feat = torch.from_numpy(compute_stereo_cqt_db(
                stereo, self.sr, n_bins=self.n_bins, 
                bins_per_octave=self.bins_per_octave, 
                hop_length=self.hop, fmin=self.fmin
            )).float()
        elif feature_name == "mfcc":
            feat = torch.from_numpy(compute_stereo_mfcc(
                stereo, self.sr, n_fft=self.n_fft, hop=self.hop, 
                win_length=self.win_length, n_mfcc=self.n_mfcc, 
                n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax, window=self.window
            )).float()
        elif feature_name == "chroma":
            feat = torch.from_numpy(compute_stereo_chroma(
                stereo, self.sr, n_fft=self.n_fft, hop=self.hop, 
                win_length=self.win_length, n_chroma=self.n_chroma, window=self.window
            )).float()
        else:
            raise ValueError(f"Unsupported feature: {feature_name}")
            
        return self._apply_norm(feat)

    def predict(self, audio_path: str | Path) -> Tuple[List[Tuple[str, float]], List[dict], np.ndarray]:
        """Runs a forward pass on an audio clip, returning global and temporal predictions."""
        stereo = load_audio_as_stereo_and_resample(Path(audio_path), target_sr=self.sr)
        
        loudness_setting = self.audio_params.get("loudness_norm", "none")
        stereo = preprocess_loudness(stereo, sr=self.sr, loudness_norm=loudness_setting)

        total_samples = stereo.shape[1]
        
        if total_samples < self.chunk_samples:
            pad_length = self.chunk_samples - total_samples
            stereo = np.pad(stereo, ((0, 0), (0, pad_length)), mode='constant')
            total_samples = self.chunk_samples

        all_chunk_features = []
        for start in range(0, total_samples, self.chunk_samples):
            end = start + self.chunk_samples
            chunk = stereo[:, start:end]
            
            if chunk.shape[1] < self.chunk_samples:
                pad_length = self.chunk_samples - chunk.shape[1]
                chunk = np.pad(chunk, ((0, 0), (0, pad_length)), mode='constant')

            feature_tensors = [self._extract_single_feature(chunk, name) for name in self.feature_names]
            all_chunk_features.append(align_and_stack_feature_tensors(feature_tensors))

        x_stacked = torch.stack(all_chunk_features).to(self.device)

        with torch.inference_mode():
            logits = self.model(x_stacked)
            
            # Get chunk-level probabilities
            if self.task_mode == "multi_label":
                chunk_probs = torch.sigmoid(logits).cpu().numpy()
            else:
                chunk_probs = torch.softmax(logits, dim=1).cpu().numpy()
                
            # Calculate global average
            global_probs = chunk_probs.mean(axis=0)

        # 1. Format Global Predictions
        global_predictions = [(self.classes[i], float(global_probs[i])) for i in range(len(self.classes))]
        global_predictions.sort(key=lambda x: x[1], reverse=True)
        
        # 2. Format Temporal Predictions (Chunk by Chunk)
        temporal_data = []
        for chunk_idx, probs in enumerate(chunk_probs):
            start_time = chunk_idx * self.clip_duration
            end_time = start_time + self.clip_duration
            
            row = {"Time Window": f"{start_time:.1f}s - {end_time:.1f}s"}
            for i, cls in enumerate(self.classes):
                row[cls] = float(probs[i])
            temporal_data.append(row)
        
        # 3. Return the feature map of the FIRST chunk for the Gradio visualisation
        vis_feature = x_stacked[0][0].cpu().numpy()
        
        return global_predictions, temporal_data, vis_feature