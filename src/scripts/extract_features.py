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
    compute_stereo_cqt_db,
    compute_stereo_mfcc,
    compute_stereo_chroma,
)

FEATURE_CHOICES = ("mel", "cqt", "mfcc", "chroma")

FEATURE_SPECS = {
    "mel": {
        "result_key": "mel_path",
        "cache_subdir": "log_mels",
        "manifest_suffix": "mels",
        "display_name": "Mel",
    },
    "cqt": {
        "result_key": "cqt_path",
        "cache_subdir": "log_cqt",
        "manifest_suffix": "cqt",
        "display_name": "CQT",
    },
    "mfcc": {
        "result_key": "mfcc_path",
        "cache_subdir": "mfcc",
        "manifest_suffix": "mfcc",
        "display_name": "MFCC",
    },
    "chroma": {
        "result_key": "chroma_path",
        "cache_subdir": "chroma",
        "manifest_suffix": "chroma",
        "display_name": "Chroma",
    },
}


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
    4. Computes the selected feature family (Mel or CQT).
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
        spec = FEATURE_SPECS[feature_type]

        if feature_type == "mel":
            feature_arr = compute_stereo_logmel_db(
                stereo,
                sr,
                n_fft=n_fft,
                hop=hop,
                win_length=win_length,
                n_mels=audio_cfg["n_mels"],
                fmin=audio_cfg["fmin"],
                fmax=audio_cfg["fmax"],
                window=window,
            )
            feature_tag = f"sr{sr}_dur{dur}_m{audio_cfg['n_mels']}_w{int(audio_cfg['win_ms'])}_{window}"
        elif feature_type == "cqt":
            feature_arr = compute_stereo_cqt_db(
                stereo,
                sr,
                n_bins=audio_cfg["n_bins"],
                bins_per_octave=audio_cfg["bins_per_octave"],
                hop_length=hop,
                fmin=audio_cfg["fmin"],
            )
            feature_tag = f"sr{sr}_dur{dur}_b{audio_cfg['n_bins']}_w{int(audio_cfg['win_ms'])}"
        elif feature_type == "mfcc":
            feature_arr = compute_stereo_mfcc(
                stereo,
                sr,
                n_fft=n_fft,
                hop=hop,
                win_length=win_length,
                n_mfcc=int(audio_cfg.get("n_mfcc", 13)),
                n_mels=int(audio_cfg.get("n_mels", 128)),
                fmin=float(audio_cfg.get("fmin", 20.0)),
                fmax=audio_cfg.get("fmax"),
                window=window,
            )
            feature_tag = (
                f"sr{sr}_dur{dur}_mfcc{int(audio_cfg.get('n_mfcc', 13))}"
                f"_m{int(audio_cfg.get('n_mels', 128))}_w{int(audio_cfg['win_ms'])}_{window}"
            )
        else:
            feature_arr = compute_stereo_chroma(
                stereo,
                sr,
                n_fft=n_fft,
                hop=hop,
                win_length=win_length,
                n_chroma=int(audio_cfg.get("n_chroma", 12)),
                window=window,
            )
            feature_tag = f"sr{sr}_dur{dur}_chroma{int(audio_cfg.get('n_chroma', 12))}_w{int(audio_cfg['win_ms'])}_{window}"

        feature_fn = f"{stem}__{hsh}__{feature_tag}.npy"
        feature_out = cache_root / spec["cache_subdir"] / label / feature_fn
        ensure_directory_exists(feature_out.parent)
        np.save(feature_out, feature_arr.astype(np.float32))
        results[spec["result_key"]] = feature_out

        return True, wav_path, label, results, None

    except Exception as e:
        return False, wav_path, label, None, str(e)

def main():
    ap = argparse.ArgumentParser(
        description="Extract Mel, CQT, MFCC, or Chroma features from a configured dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.scripts.extract_features --dataset irmas --feature cqt\n"
            "  python -m src.scripts.extract_features --dataset chinese_instruments --feature mfcc --num_workers 8\n"
        ),
    )
    ap.add_argument("--config", default="src/configs/audio_params.yaml", help="Path to YAML audio/dataset config.")
    ap.add_argument("--dataset", required=True, help="Dataset key under 'datasets' in config (e.g., irmas).")
    ap.add_argument(
        "--feature",
        required=True,
        help="Feature family: mel | cqt | mfcc | chroma",
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
    if args.feature == "cqt":
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

    rows_by_feature = {name: [] for name in FEATURE_CHOICES}
    n_success = 0
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
                n_success += 1
                for feature_name, spec in FEATURE_SPECS.items():
                    result_key = spec["result_key"]
                    if result_key in results:
                        rows_by_feature[feature_name].append([str(results[result_key]), label, str(wav_path)])
            else:
                n_fail += 1
                print(f"\n[ERROR] {wav_path.name}: {err}")

    # --- Stage 3: Manifest Generation ---
    # We follow the format: filepath (npy), label, wavpath (source)
    
    feature_rows = rows_by_feature.get(args.feature, [])
    if feature_rows:
        spec = FEATURE_SPECS[args.feature]
        manifest_path = manifest_base.with_name(f"{args.dataset}_train_{spec['manifest_suffix']}.csv")
        ensure_directory_exists(manifest_path.parent)
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "label", "wavpath"])
            writer.writerows(feature_rows)
        print(f"Saved {spec['display_name']} manifest: {manifest_path}")

    print(f"Extraction complete. Success: {n_success}, Fail: {n_fail}")

if __name__ == "__main__":
    main()
