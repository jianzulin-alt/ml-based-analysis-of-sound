from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm

from src.preprocessing.features import (
    compute_stft_params,
    ensure_duration,
    load_audio_as_stereo,
    compute_stereo_logmel_db,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_and_preprocess(path: Path | str, cfg: dict) -> torch.Tensor:
    """Load audio and convert to the 2-channel Log-Mel tensor expected by the model."""
    wav_path = Path(path)
    stereo = load_audio_as_stereo(wav_path, target_sr=cfg["sr"])
    stereo = ensure_duration(stereo, cfg["sr"], cfg["duration"])
    n_fft, hop, win_length = compute_stft_params(cfg["sr"], cfg["win_ms"], cfg["hop_ms"])

    mel = compute_stereo_logmel_db(
        stereo,
        cfg["sr"],
        n_fft=n_fft,
        hop=hop,
        win_length=win_length,
        n_mels=cfg["n_mels"],
        fmin=cfg["fmin"],
        fmax=cfg.get("fmax"),
        window=cfg.get("window", "hann"),
    )
    return torch.from_numpy(mel).float()


def get_prediction(model: torch.nn.Module, mel: torch.Tensor, device: torch.device) -> np.ndarray:
    """Apply model normalisation and return probability vector."""
    model.eval()
    with torch.no_grad():
        x = mel.unsqueeze(0).to(device)
        x = (x - x.mean()) / (x.std() + 1e-6)
        logits = model(x)
        probs = torch.sigmoid(logits)
    return probs.cpu().numpy()[0]


def _resolve_wav_paths(wav_paths: Iterable[Path | str], root: Path) -> list[Path]:
    resolved_paths: list[Path] = []
    for wav_path in wav_paths:
        path = Path(wav_path)
        resolved = path if path.is_absolute() else (root / path)
        resolved = resolved.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"WAV file not found: {resolved}")
        resolved_paths.append(resolved)
    return resolved_paths


def run_inference_on_wav_files(
    *,
    model_cls,
    model_kwargs: dict,
    model_weights_path,
    device,
    wav_paths: Iterable[Path | str],
    root: Path | None = None,
    state_key: str = "model_state",
    audio_cfg_key: str = "audio_config",
    classes_key: str = "classes",
    strict_load: bool = True,
    show_progress: bool = True,
):
    """
    Load checkpoint + model, then run inference directly on WAV paths.

    Returns:
        preds_arr: (N, C) float array of predicted probabilities
        sample_ids: list[str] stems of wav filenames
        resolved_wav_paths: list[Path] fully resolved wav paths
        audio_cfg: dict-like audio config from checkpoint
        valid_labels: list[str] normalised class names
    """
    if not isinstance(device, torch.device):
        device = torch.device(device)

    ckpt = torch.load(model_weights_path, map_location=device)

    audio_cfg = ckpt[audio_cfg_key]
    valid_labels = [c.strip().lower() for c in ckpt[classes_key]]

    model = model_cls(**model_kwargs, num_classes=len(valid_labels)).to(device)
    model.load_state_dict(ckpt[state_key], strict=strict_load)
    model.eval()

    base_root = PROJECT_ROOT if root is None else Path(root)
    resolved_wav_paths = _resolve_wav_paths(wav_paths, base_root)

    if not resolved_wav_paths:
        raise ValueError("No WAV paths provided for inference.")

    all_preds: list[np.ndarray] = []
    sample_ids: list[str] = []

    print(
        f"Running inference on {len(resolved_wav_paths)} WAV files against {len(valid_labels)} classes..."
    )

    iterator = resolved_wav_paths
    if show_progress:
        iterator = tqdm(resolved_wav_paths, total=len(resolved_wav_paths))

    with torch.no_grad():
        for wav_path in iterator:
            mel = load_and_preprocess(wav_path, audio_cfg)
            probs = get_prediction(model, mel, device)
            all_preds.append(probs)
            sample_ids.append(wav_path.stem)

    preds_arr = np.asarray(all_preds)
    return preds_arr, sample_ids, resolved_wav_paths, audio_cfg, valid_labels
