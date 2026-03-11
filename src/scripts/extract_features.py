#!/usr/bin/env python3
"""
Extract Features Utility
------------------------
Orchestrates the conversion of raw audio datasets into DSP feature tensors (.npy).
Supports:
- Log-Mel Spectrograms
- Constant-Q Transform (CQT)
- Multiprocessing for high-volume datasets (IRMAS, Chinese Instruments)
- Automatic CSV manifest generation compatible with the model trainer
"""

import argparse
import csv
import math
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Sequence, Set

import yaml
import numpy as np
from tqdm import tqdm


from src.preprocessing.preprocessing import (
    load_audio_as_stereo_and_resample, 
    conform_audio_duration, 
    preprocess_loudness,
    generate_path_hash,
    ensure_directory_exists
)
from src.preprocessing.features import (
    compute_stft_params, 
    compute_stereo_logmel_db, 
    compute_stereo_cqt_db
)

FEATURE_CHOICES = ("mel", "cqt", "mel_cqt")


def parse_dataset(dataset_dir: Path):
    """
    Scans a directory for .wav files.
    Assumes a folder-per-class structure: <dataset_dir>/<label>/<file>.wav
    Returns: Generator yielding (Path, label_string)
    """
    for wav_path in dataset_dir.rglob("*.wav"):
        if wav_path.is_file():
            # The immediate parent folder name is treated as the class label
            yield wav_path, wav_path.parent.name.strip().lower()


def max_safe_cqt_bins(sr: int, fmin: float, bins_per_octave: int) -> int:
    """
    Return the largest valid CQT bin count that keeps top frequency <= Nyquist.
    """
    if sr <= 0:
        raise ValueError("sample rate (sr) must be > 0")
    if fmin <= 0:
        raise ValueError("fmin must be > 0")
    if bins_per_octave <= 0:
        raise ValueError("bins_per_octave must be > 0")

    nyquist = sr / 2.0
    if fmin >= nyquist:
        return 1

    return int(math.floor(bins_per_octave * math.log2(nyquist / fmin))) + 1


def normalize_feature_name(feature: str) -> str:
    raw = str(feature).strip().lower()
    if raw not in FEATURE_CHOICES:
        allowed = " | ".join(FEATURE_CHOICES)
        raise ValueError(f"Unsupported feature '{feature}'. Choose one of: {allowed}")
    return raw


def load_allowed_labels(
    labels_config: Optional[Path],
    dataset_key: str,
    label_key: Optional[str],
) -> Optional[Set[str]]:
    if labels_config is None:
        return None
    if not labels_config.exists():
        print(f"[WARN] labels config not found: {labels_config}. Using all labels from folders.")
        return None

    with open(labels_config, "r", encoding="utf-8") as f:
        labels_cfg = yaml.safe_load(f) or {}

    keys_to_try = []
    if label_key:
        keys_to_try.append(label_key)
    if dataset_key == "irmas":
        keys_to_try.extend(["irmas_labels", "train_labels"])
    else:
        keys_to_try.extend(["train_labels", "labels"])

    labels: Optional[Sequence[str]] = None
    for key in keys_to_try:
        maybe = labels_cfg.get(key)
        if isinstance(maybe, (list, tuple)):
            labels = maybe
            break

    if not labels:
        print(
            f"[WARN] No label list found in {labels_config} for keys {keys_to_try}. "
            "Using all labels from folders."
        )
        return None

    return {str(x).strip().lower() for x in labels if x is not None}

def _process_one(wav_path: Path, label: str, cache_root: Path, audio_cfg: dict, feature_type: str):
    """
    Worker function for parallel execution.
    1. Loads and resamples audio.
    2. Enforces duration (padding/cropping).
    3. Normalises loudness (LUFS).
    4. Computes Mel and/or CQT.
    5. Saves to disk with a collision-resistant hash.
    """
    try:
        sr = int(audio_cfg["sr"])
        dur = float(audio_cfg["duration"])
        
        # --- Stage 1: Waveform Manipulation ---
        stereo = load_audio_as_stereo_and_resample(wav_path, target_sr=sr)
        stereo = conform_audio_duration(stereo, sr, dur)
        stereo = preprocess_loudness(
            stereo, sr=sr, 
            loudness_norm=audio_cfg.get("loudness_norm", "none"),
            target_lufs=audio_cfg.get("target_lufs", -23.0),
            peak_limit=audio_cfg.get("loudness_peak_limit", 0.99)
        )

        # Calculate FFT/Hop sizes from millisecond configurations
        n_fft, hop, win_length = compute_stft_params(sr, audio_cfg["win_ms"], audio_cfg["hop_ms"])
        window = str(audio_cfg.get("window", "hann"))
        
        stem = wav_path.stem
        # Generate hash of path to ensure unique filenames if different sources have same stem
        hsh = generate_path_hash(str(wav_path))
        results = {}

        # --- Stage 2: Feature Extraction ---
        
        # Log-Mel Extraction
        if feature_type in {"mel", "mel_cqt"}:
            mel = compute_stereo_logmel_db(
                stereo, sr, n_fft=n_fft, hop=hop, win_length=win_length,
                n_mels=audio_cfg["n_mels"], fmin=audio_cfg["fmin"], fmax=audio_cfg["fmax"],
                window=window,
            )
            # Tag filename with params to prevent cache invalidation issues
            mel_tag = f"sr{sr}_dur{dur}_m{audio_cfg['n_mels']}_w{int(audio_cfg['win_ms'])}_{window}"
            mel_fn = f"{stem}__{hsh}__{mel_tag}.npy"
            mel_out = cache_root / "log_mels" / label / mel_fn
            
            ensure_directory_exists(mel_out.parent)
            np.save(mel_out, mel.astype(np.float32))
            results["mel_path"] = mel_out

        # CQT Extraction
        if feature_type in {"cqt", "mel_cqt"}:
            cqt = compute_stereo_cqt_db(
                stereo, sr, n_bins=audio_cfg["n_bins"], 
                bins_per_octave=audio_cfg["bins_per_octave"], 
                hop_length=hop, fmin=audio_cfg["fmin"]
            )
            cqt_tag = f"sr{sr}_dur{dur}_b{audio_cfg['n_bins']}_w{int(audio_cfg['win_ms'])}"
            cqt_fn = f"{stem}__{hsh}__{cqt_tag}.npy"
            cqt_out = cache_root / "log_cqt" / label / cqt_fn
            
            ensure_directory_exists(cqt_out.parent)
            np.save(cqt_out, cqt.astype(np.float32))
            results["cqt_path"] = cqt_out

        return True, wav_path, label, results, None

    except Exception as e:
        return False, wav_path, label, None, str(e)

def main():
    ap = argparse.ArgumentParser(
        description="Extract Mel and/or CQT features from a configured dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  .venv/bin/python src/scripts/extract_features.py --dataset irmas --feature mel_cqt\n"
            "  .venv/bin/python src/scripts/extract_features.py --dataset chinese_instruments --feature mel --num_workers 8\n"
        ),
    )
    ap.add_argument("--config", default="src/configs/audio_params.yaml", help="Path to YAML audio/dataset config.")
    ap.add_argument("--dataset", required=True, help="Dataset key under 'datasets' in config (e.g., irmas).")
    ap.add_argument(
        "--feature",
        required=True,
        help="Feature family: mel | cqt | mel_cqt ",
    )
    ap.add_argument("--num_workers", type=int, default=12, help="Parallel worker processes.")
    ap.add_argument(
        "--labels_config",
        default="src/configs/labels.yaml",
        help="Optional labels YAML used to filter extracted labels.",
    )
    ap.add_argument("--label_key", default=None, help="Optional key inside labels YAML (e.g. irmas_labels).")
    args = ap.parse_args()
    try:
        args.feature = normalize_feature_name(args.feature)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(2)

    # Load configuration
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    if args.dataset not in cfg["datasets"]:
        print(f"Error: Dataset '{args.dataset}' not defined in config.")
        sys.exit(1)

    dataset_cfg = cfg["datasets"][args.dataset]
    audio_cfg = cfg["audio"]
    labels_config = Path(args.labels_config).expanduser() if args.labels_config else None
    allowed_labels = load_allowed_labels(labels_config, args.dataset, args.label_key)

    # Guard CQT settings so librosa.cqt does not exceed Nyquist.
    if args.feature in {"cqt", "mel_cqt"}:
        sr = int(audio_cfg["sr"])
        fmin = float(audio_cfg["fmin"])
        bins_per_octave = int(audio_cfg["bins_per_octave"])
        requested_n_bins = int(audio_cfg["n_bins"])
        safe_n_bins = max_safe_cqt_bins(sr, fmin, bins_per_octave)
        if requested_n_bins > safe_n_bins:
            print(
                "[WARN] Requested CQT n_bins="
                f"{requested_n_bins} exceeds Nyquist-safe limit ({safe_n_bins}) "
                f"for sr={sr}, fmin={fmin}, bins_per_octave={bins_per_octave}. "
                f"Using n_bins={safe_n_bins}."
            )
            audio_cfg = dict(audio_cfg)
            audio_cfg["n_bins"] = safe_n_bins
    
    train_dir = Path(dataset_cfg["train_dir"])
    cache_root = Path(dataset_cfg["cache_root"])
    # Resolve absolute path for manifest to ensure CSV is portable
    manifest_base = Path(dataset_cfg["manifest"]).resolve()
    
    # Gather files
    wavs_and_labels = [
        (wav_path, label)
        for wav_path, label in parse_dataset(train_dir)
        if allowed_labels is None or label in allowed_labels
    ]
    if not wavs_and_labels:
        print(f"No audio files found in {train_dir}")
        return

    if allowed_labels is not None:
        present_labels = {label for _, label in wavs_and_labels}
        missing_labels = sorted(allowed_labels - present_labels)
        print(
            f"Filtering to {len(allowed_labels)} configured labels. "
            f"Found {len(present_labels)} labels with audio under {train_dir}."
        )
        if missing_labels:
            print(f"[WARN] Configured labels with no audio folders/files: {', '.join(missing_labels)}")

    print(f"Processing {len(wavs_and_labels)} files from {args.dataset}...")

    mel_rows, cqt_rows = [], []
    n_fail = 0

    # Multi-core processing
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(_process_one, wav, lbl, cache_root, audio_cfg, args.feature): wav
            for wav, lbl in wavs_and_labels
        }
        
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"DSP: {args.dataset}"):
            success, wav_path, label, results, err = fut.result()
            
            if success:
                # Store paths as strings for CSV writing
                if "mel_path" in results:
                    mel_rows.append([str(results["mel_path"]), label, str(wav_path)])
                if "cqt_path" in results:
                    cqt_rows.append([str(results["cqt_path"]), label, str(wav_path)])
            else:
                n_fail += 1
                print(f"\n[ERROR] {wav_path.name}: {err}")

    # --- Stage 3: Manifest Generation ---
    # We follow the format: filepath (npy), label, wavpath (source)
    
    if mel_rows:
        mel_csv = manifest_base.with_name(f"{args.dataset}_train_mels.csv")
        ensure_directory_exists(mel_csv.parent)
        with open(mel_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "label", "wavpath"])
            writer.writerows(mel_rows)
        print(f"Saved Mel manifest: {mel_csv}")

    if cqt_rows:
        cqt_csv = manifest_base.with_name(f"{args.dataset}_train_cqt.csv")
        ensure_directory_exists(cqt_csv.parent)
        with open(cqt_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "label", "wavpath"])
            writer.writerows(cqt_rows)
        print(f"Saved CQT manifest: {cqt_csv}")

    print(f"Extraction complete. Success: {len(mel_rows) or len(cqt_rows)}, Fail: {n_fail}")

if __name__ == "__main__":
    main()
