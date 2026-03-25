#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm

from src.utils.system_utils import load_yaml
from src.utils.audio_utils import parse_dataset, max_safe_cqt_bins, load_allowed_labels
from src.preprocessing.audio_io import (
    load_audio_as_stereo_and_resample, conform_audio_duration, 
    preprocess_loudness, generate_path_hash, ensure_directory_exists
)
from src.preprocessing.features import (
    compute_stft_params, compute_stereo_logmel_db, compute_stereo_cqt_db,
    compute_stereo_mfcc, compute_stereo_chroma
)

FEATURE_CHOICES = ("mel", "cqt", "mfcc", "chroma")
FEATURE_SPECS = {
    "mel": {"result_key": "mel_path", "cache_subdir": "log_mels", "manifest_suffix": "mels", "display_name": "Mel"},
    "cqt": {"result_key": "cqt_path", "cache_subdir": "log_cqt", "manifest_suffix": "cqt", "display_name": "CQT"},
    "mfcc": {"result_key": "mfcc_path", "cache_subdir": "mfcc", "manifest_suffix": "mfcc", "display_name": "MFCC"},
    "chroma": {"result_key": "chroma_path", "cache_subdir": "chroma", "manifest_suffix": "chroma", "display_name": "Chroma"},
}

def normalize_feature(arr: np.ndarray, norm_type: str) -> np.ndarray:
    """Applies normalisation to the feature array."""
    if norm_type == "min_max":
        a_min, a_max = arr.min(), arr.max()
        if a_max - a_min > 1e-8:
            return (arr - a_min) / (a_max - a_min)
        return arr
    elif norm_type == "standard":
        mean, std = arr.mean(), arr.std()
        if std > 1e-8:
            return (arr - mean) / std
        return arr - mean
    return arr

def _process_one(wav_path: Path, label: str, cache_root: Path, audio_cfg: dict, feature_type: str):
    extracted_files = []
    try:
        sr = int(audio_cfg["sr"])
        dur = float(audio_cfg["duration"])
        
        # Load and standardise audio
        stereo = load_audio_as_stereo_and_resample(wav_path, target_sr=sr)
        stereo = conform_audio_duration(stereo, sr, dur)
        stereo = preprocess_loudness(
            stereo, sr=sr, loudness_norm=audio_cfg.get("loudness_norm", "none"),
            target_lufs=audio_cfg.get("target_lufs", -23.0), peak_limit=audio_cfg.get("loudness_peak_limit", 0.99)
        )

        n_fft, hop, win_length = compute_stft_params(sr, audio_cfg["win_ms"], audio_cfg["hop_ms"])
        window = str(audio_cfg.get("window", "hann"))
        hsh = generate_path_hash(str(wav_path))
        spec = FEATURE_SPECS[feature_type]
        norm_type = audio_cfg.get("feature_norm", "none")

        # Set up combinations for augmentation
        audio_variations = [("orig", stereo)]
        if audio_cfg.get("channel_swap_augmentation", False):
            # Reverse the channel axis to swap left/right
            audio_variations.append(("swapped", stereo[::-1, :]))

        for aug_tag, audio_data in audio_variations:
            if feature_type == "mel":
                feature_arr = compute_stereo_logmel_db(audio_data, sr, n_fft=n_fft, hop=hop, win_length=win_length, n_mels=audio_cfg["n_mels"], fmin=audio_cfg["fmin"], fmax=audio_cfg["fmax"], window=window)
                feature_tag = f"sr{sr}_dur{dur}_m{audio_cfg['n_mels']}_w{int(audio_cfg['win_ms'])}_{window}_{aug_tag}"
            elif feature_type == "cqt":
                feature_arr = compute_stereo_cqt_db(audio_data, sr, n_bins=audio_cfg["n_bins"], bins_per_octave=audio_cfg["bins_per_octave"], hop_length=hop, fmin=audio_cfg["fmin"])
                feature_tag = f"sr{sr}_dur{dur}_b{audio_cfg['n_bins']}_w{int(audio_cfg['win_ms'])}_{aug_tag}"
            elif feature_type == "mfcc":
                feature_arr = compute_stereo_mfcc(audio_data, sr, n_fft=n_fft, hop=hop, win_length=win_length, n_mfcc=int(audio_cfg.get("n_mfcc", 13)), n_mels=int(audio_cfg.get("n_mels", 128)), fmin=float(audio_cfg.get("fmin", 20.0)), fmax=audio_cfg["fmax"], window=window)
                feature_tag = f"sr{sr}_dur{dur}_mfcc_m{int(audio_cfg.get('n_mels', 128))}_w{int(audio_cfg['win_ms'])}_{window}_{aug_tag}"
            else:
                feature_arr = compute_stereo_chroma(audio_data, sr, n_fft=n_fft, hop=hop, win_length=win_length, n_chroma=int(audio_cfg.get("n_chroma", 12)), window=window)
                feature_tag = f"sr{sr}_dur{dur}_chroma_w{int(audio_cfg['win_ms'])}_{window}_{aug_tag}"

            # Apply normalisation
            feature_arr = normalize_feature(feature_arr, norm_type)

            feature_out = cache_root / spec["cache_subdir"] / label / f"{wav_path.stem}__{hsh}__{feature_tag}.npy"
            ensure_directory_exists(feature_out.parent)
            np.save(feature_out, feature_arr.astype(np.float32))
            
            extracted_files.append((True, wav_path, label, {spec["result_key"]: feature_out}, None))

        return extracted_files

    except Exception as e:
        return [(False, wav_path, label, None, str(e))]

def main():
    ap = argparse.ArgumentParser(description="Extract DSP features.")
    ap.add_argument("--config", default="src/configs/audio_params.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--feature", required=True, choices=FEATURE_CHOICES)
    ap.add_argument("--num_workers", type=int, default=12)
    ap.add_argument("--labels_config", default="src/configs/labels.yaml")
    ap.add_argument("--label_key", default=None)
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    dataset_cfg = cfg["datasets"][args.dataset]
    audio_cfg = cfg["audio"]
    
    if args.feature == "cqt":
        safe_bins = max_safe_cqt_bins(audio_cfg["sr"], audio_cfg["fmin"], audio_cfg["bins_per_octave"])
        audio_cfg["n_bins"] = min(audio_cfg["n_bins"], safe_bins)

    allowed_labels = load_allowed_labels(Path(args.labels_config), args.dataset, args.label_key)
    wavs_and_labels = [(w, l) for w, l in parse_dataset(Path(dataset_cfg["train_dir"])) if not allowed_labels or l in allowed_labels]

    rows = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(_process_one, w, l, Path(dataset_cfg["cache_root"]), audio_cfg, args.feature): w for w, l in wavs_and_labels}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="DSP Extract"):
            results = fut.result()
            for success, w, l, res, err in results:
                if success:
                    rows.append([str(res[FEATURE_SPECS[args.feature]["result_key"]]), l, str(w)])
                else:
                    print(f"\n[ERROR] Failed to process {w}: {err}")

    manifest_path = Path(dataset_cfg["manifest"]).resolve().with_name(f"{args.dataset}_train_{FEATURE_SPECS[args.feature]['manifest_suffix']}.csv")
    ensure_directory_exists(manifest_path.parent)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "wavpath"])
        writer.writerows(rows)
    print(f"Extraction complete. Manifest saved to: {manifest_path}")

if __name__ == "__main__":
    main()