from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.utils.system_utils import get_repo_root, resolve_path, load_yaml
from src.utils.train_utils import get_device
from src.models.builder import build_model
from src.preprocessing.feature_modes import (
    align_and_stack_feature_tensors, 
    feature_mode_to_features, 
    feature_mode_to_in_channels, 
    normalize_feature_mode
)
from src.preprocessing.features import (
    compute_stereo_cqt_db, 
    compute_stereo_logmel_db, 
    compute_stereo_mfcc, 
    compute_stereo_chroma, 
    compute_stft_params
)
from src.preprocessing.audio_io import (
    load_audio_as_stereo_and_resample, 
    preprocess_loudness
)

@dataclass
class TestSample:
    wav_path: Path
    txt_path: Path
    labels: list[str]  # Changed to a list to support polyphonic ground truth

def _read_txt_labels(path: Path) -> list[str]:
    """Reads ALL valid labels from the accompanying text file."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    labels = [line.strip().lower().split()[0] for line in raw.splitlines() if line.strip()]
    return list(set(labels)) # Return unique labels

def collect_test_samples(test_root: Path, valid_labels_set: set[str]) -> list[TestSample]:
    """Scans the test directory for wav files and their corresponding label text files."""
    samples = []
    for wav_path in sorted(test_root.rglob("*.wav")):
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists(): 
            continue
        
        labels = _read_txt_labels(txt_path)
        # Keep only the labels that are in our model's classes
        valid_labels = [l for l in labels if l in valid_labels_set]
        
        if valid_labels:
            samples.append(TestSample(wav_path, txt_path, valid_labels))
    return samples

class SlidingWindowTestDataset(Dataset):
    """
    Dynamically slices long test audio into model-sized chunks (e.g., 3s)
    and extracts properly normalised features for every chunk.
    """
    def __init__(self, samples: list[TestSample], classes: list[str], audio_cfg: dict, feature_mode: str) -> None:
        self.samples = samples
        self.classes = [c.strip().lower() for c in classes]
        self.label_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.feature_mode = normalize_feature_mode(feature_mode)
        self.feature_names = feature_mode_to_features(self.feature_mode)

        self.sr = int(audio_cfg["sr"])
        self.duration = float(audio_cfg["duration"])
        self.chunk_samples = int(self.duration * self.sr)
        
        self.n_fft, self.hop, self.win_length = compute_stft_params(
            self.sr, float(audio_cfg["win_ms"]), float(audio_cfg["hop_ms"])
        )
        self.window = str(audio_cfg.get("window", "hann"))
        self.fmin = float(audio_cfg.get("fmin", 20.0))
        self.fmax = float(audio_cfg.get("fmax", self.sr / 2))
        self.n_mels = int(audio_cfg.get("n_mels", 128))
        self.n_bins = int(audio_cfg.get("n_bins", 120))
        self.bins_per_octave = int(audio_cfg.get("bins_per_octave", 12))
        self.n_mfcc = int(audio_cfg.get("n_mfcc", 13))
        self.n_chroma = int(audio_cfg.get("n_chroma", 12))
        
        self.feature_norm = str(audio_cfg.get("feature_norm", "none")).lower()

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_norm == "min_max":
            x_min, x_max = x.min(), x.max()
            if x_max - x_min > 1e-8: return (x - x_min) / (x_max - x_min)
        elif self.feature_norm == "standard":
            return (x - x.mean()) / (x.std() + 1e-8)
        return x

    def _extract_single_feature(self, stereo: np.ndarray, feature_name: str) -> torch.Tensor:
        if feature_name == "mel":
            feat = torch.from_numpy(compute_stereo_logmel_db(stereo, self.sr, n_fft=self.n_fft, hop=self.hop, win_length=self.win_length, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax, window=self.window)).float()
        elif feature_name == "cqt":
            feat = torch.from_numpy(compute_stereo_cqt_db(stereo, self.sr, n_bins=self.n_bins, bins_per_octave=self.bins_per_octave, hop_length=self.hop, fmin=self.fmin)).float()
        elif feature_name == "mfcc":
            feat = torch.from_numpy(compute_stereo_mfcc(stereo, self.sr, n_fft=self.n_fft, hop=self.hop, win_length=self.win_length, n_mfcc=self.n_mfcc, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax, window=self.window)).float()
        elif feature_name == "chroma":
            feat = torch.from_numpy(compute_stereo_chroma(stereo, self.sr, n_fft=self.n_fft, hop=self.hop, win_length=self.win_length, n_chroma=self.n_chroma, window=self.window)).float()
        else:
            raise ValueError(f"Unsupported feature: {feature_name}")
        return self._apply_norm(feat)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        stereo = load_audio_as_stereo_and_resample(sample.wav_path, target_sr=self.sr)
        stereo = preprocess_loudness(stereo, sr=self.sr, loudness_norm="none")

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

        # Create multi-hot target
        target = torch.zeros(len(self.classes), dtype=torch.float32)
        for label in sample.labels:
            target[self.label_to_idx[label]] = 1.0

        x_stacked = torch.stack(all_chunk_features)
        return x_stacked, sample.wav_path.name, target

def evaluate_dataset(
    model_path: str | Path,
    dataset_name: Optional[str] = None,
    test_path: Optional[str | Path] = None,
    *,
    audio_config_path: str | Path = "src/configs/audio_params.yaml",
    num_workers: int = 4,
    lenient_eval: bool = False, # NEW PARAMETER
) -> dict[str, Any]:
    
    root = get_repo_root()
    model_resolved = resolve_path(model_path, root)
    ckpt_path = model_resolved if model_resolved.is_file() else model_resolved / "best_val.pt"
    run_dir = ckpt_path.parent
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    run_cfg = load_yaml(run_dir / "run_config.yaml") if (run_dir / "run_config.yaml").exists() else ckpt.get("config", {})
    
    task_mode = run_cfg.get("task_mode", "single_label")
    audio_params = run_cfg.get("audio_params", load_yaml(resolve_path(audio_config_path, root)).get("audio", {}))
        
    classes = run_cfg.get("classes", [])
    if not classes: raise ValueError("Could not find a 'classes' list in run_config.yaml.")
        
    feature_mode = normalize_feature_mode(run_cfg.get("feature_mode", "mel"))
    model_cfg = run_cfg.get("model", {})
    
    if test_path: test_root = resolve_path(test_path, root)
    elif dataset_name: test_root = resolve_path(load_yaml(resolve_path(audio_config_path, root)).get("datasets", {}).get(dataset_name, {}).get("test", ""), root)
    else: raise ValueError("Provide dataset_name or test_path.")

    samples = collect_test_samples(test_root, set(classes))
    device_str, _, _, _, pin_mem = get_device()
    infer_device = torch.device(device_str)
    
    model = build_model(model_cfg.get("backbone", "cnn"), feature_mode_to_in_channels(feature_mode), len(classes), model_cfg)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=True)
    model.to(infer_device).eval()

    ds = SlidingWindowTestDataset(samples, classes, audio_params, feature_mode)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

    all_preds, all_targets, filenames = [], [], []

    with torch.no_grad():
        for x_stacked, names, y_true_multi in tqdm(dl, desc=f"Evaluating {test_root.name} (Lenient: {lenient_eval})"):
            x_chunks = x_stacked.squeeze(0).to(infer_device, non_blocking=pin_mem)
            logits = model(x_chunks)
            
            true_multi = y_true_multi[0].numpy() # Multi-hot vector

            if task_mode == "multi_label":
                probs = torch.sigmoid(logits).mean(dim=0, keepdim=True).cpu()
                preds = (probs > 0.5).int()[0].numpy()
                all_preds.append(preds)
                all_targets.append(true_multi)
                
            else: # Single Label Evaluation
                probs = torch.softmax(logits, dim=1).mean(dim=0, keepdim=True).cpu()
                pred_idx = torch.argmax(probs, dim=1).item()
                
                # --- LENIENT EVALUATION LOGIC ---
                # This handles polyphonic test sets where multiple instruments are present.
                # If 'lenient_eval' is True, we count the prediction as a success if the 
                # predicted instrument exists ANYWHERE in the ground-truth label list 
                # for this specific audio file.
                if lenient_eval:
                    if true_multi[pred_idx] == 1.0:
                        # Correct! We pretend the predicted label was the definitive ground truth
                        effective_true = pred_idx
                    else:
                        # Incorrect. Fall back to the first available true label for the Confusion Matrix
                        effective_true = np.argmax(true_multi)
                else:
                    # Strict: Always compare against the first label listed in the file
                    effective_true = np.argmax(true_multi)
                    
                all_preds.append(pred_idx)
                all_targets.append(effective_true)
                
            filenames.extend(names)

    # Scikit-learn handles both 1D (single-label) and 2D (multi-label) arrays perfectly
    report = classification_report(all_targets, all_preds, target_names=classes, output_dict=True, zero_division=0)
    
    cm = None
    if task_mode == "single_label":
        cm = confusion_matrix(all_targets, all_preds)
    
    return {
        "task_mode": task_mode,
        "classification_report": report,
        "confusion_matrix": cm,
        "classes": classes
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--model_dir", required=True, help="Path to the saved model directory")
    parser.add_argument("--dataset", choices=["irmas", "chinese_instruments"], help="Dataset key from audio_params.yaml")
    parser.add_argument("--test_path", help="Direct path to an audio folder for evaluation")
    parser.add_argument("--lenient", action="store_true", help="Count prediction as correct if it matches ANY ground-truth label in the file.")
    args = parser.parse_args()
    
    results = evaluate_dataset(args.model_dir, dataset_name=args.dataset, test_path=args.test_path, lenient_eval=args.lenient)
    print(f"\nOverall Accuracy: {results['classification_report']['accuracy']:.4f}")