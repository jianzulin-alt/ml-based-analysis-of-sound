#!/usr/bin/env python3
"""
Generate synthetic polyphonic mixtures and extract Mel/CQT features.

This script is designed for multi-label training. It creates mixtures from
single-label source clips, then writes manifests with a `labels` column
containing pipe-separated labels (e.g. "dizi|pipa").
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import yaml
from tqdm import tqdm

from src.preprocessing.preprocessing import (
    conform_audio_duration,
    ensure_directory_exists,
    generate_path_hash,
    load_audio_as_stereo_and_resample,
    preprocess_loudness,
)
from src.preprocessing.features import (
    compute_stereo_cqt_db,
    compute_stereo_logmel_db,
    compute_stft_params,
)

EPS = 1e-8
FEATURE_CHOICES = ("mel", "cqt", "mel_cqt")


def parse_dataset(dataset_dir: Path) -> Iterable[Tuple[Path, str]]:
    for wav_path in dataset_dir.rglob("*.wav"):
        if wav_path.is_file():
            yield wav_path, wav_path.parent.name.strip().lower()


def max_safe_cqt_bins(sr: int, fmin: float, bins_per_octave: int) -> int:
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

    keys_to_try: List[str] = []
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


def build_label_index(train_dir: Path, allowed_labels: Optional[Set[str]]) -> Dict[str, List[Path]]:
    label_to_wavs: Dict[str, List[Path]] = defaultdict(list)
    for wav_path, label in parse_dataset(train_dir):
        if allowed_labels is not None and label not in allowed_labels:
            continue
        label_to_wavs[label].append(wav_path)

    return {label: paths for label, paths in label_to_wavs.items() if paths}


def rms(stereo: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(stereo), dtype=np.float64) + EPS))


def mix_sources(sources: Sequence[np.ndarray], snr_db: Sequence[float], peak_limit: float) -> np.ndarray:
    if not sources:
        raise ValueError("No source signals provided for mixing")

    mix = sources[0].astype(np.float32).copy()
    base_rms = rms(mix)

    for src, snr in zip(sources[1:], snr_db):
        src_rms = rms(src)
        # Positive SNR means base source is louder than added source.
        target_ratio = 10.0 ** (-float(snr) / 20.0)
        gain = (base_rms * target_ratio) / max(src_rms, EPS)
        mix += src.astype(np.float32) * gain

    peak = float(np.max(np.abs(mix)))
    if peak_limit > 0 and peak > peak_limit:
        mix *= (peak_limit / peak)

    return mix.astype(np.float32)


def build_mix_plan(
    label_to_wavs: Dict[str, List[Path]],
    *,
    num_mixes: int,
    min_sources: int,
    max_sources: int,
    snr_db_min: float,
    snr_db_max: float,
    seed: int,
) -> List[dict]:
    labels = sorted(label_to_wavs.keys())
    rng = np.random.default_rng(seed)
    plans: List[dict] = []

    for i in range(num_mixes):
        n_src = int(rng.integers(min_sources, max_sources + 1))
        chosen_labels = rng.choice(labels, size=n_src, replace=False).tolist()

        chosen_paths: List[str] = []
        for label in chosen_labels:
            options = label_to_wavs[label]
            wav_idx = int(rng.integers(0, len(options)))
            chosen_paths.append(str(options[wav_idx]))

        snr_values = [float(rng.uniform(snr_db_min, snr_db_max)) for _ in range(max(0, n_src - 1))]
        plans.append(
            {
                "mix_id": f"mix_{i:06d}",
                "labels": chosen_labels,
                "wav_paths": chosen_paths,
                "snr_db": snr_values,
            }
        )

    return plans


def _process_one_mix(plan: dict, cache_root: Path, audio_cfg: dict, feature_type: str):
    try:
        sr = int(audio_cfg["sr"])
        dur = float(audio_cfg["duration"])
        peak_limit = float(audio_cfg.get("loudness_peak_limit", 0.99))

        sources: List[np.ndarray] = []
        for wav_str in plan["wav_paths"]:
            wav_path = Path(wav_str)
            stereo = load_audio_as_stereo_and_resample(wav_path, target_sr=sr)
            stereo = conform_audio_duration(stereo, sr, dur)
            stereo = preprocess_loudness(
                stereo,
                sr=sr,
                loudness_norm=audio_cfg.get("loudness_norm", "none"),
                target_lufs=audio_cfg.get("target_lufs", -23.0),
                peak_limit=peak_limit,
            )
            sources.append(stereo)

        mixed = mix_sources(sources, plan["snr_db"], peak_limit=peak_limit)
        n_fft, hop, win_length = compute_stft_params(sr, audio_cfg["win_ms"], audio_cfg["hop_ms"])
        window = str(audio_cfg.get("window", "hann"))

        mix_id = str(plan["mix_id"])
        labels_str = "|".join(sorted(set(str(x).strip().lower() for x in plan["labels"])))
        sources_str = "|".join(plan["wav_paths"])
        src_hash = generate_path_hash(sources_str)

        row = {
            "mix_id": mix_id,
            "labels": labels_str,
            "sources": sources_str,
        }

        if feature_type in {"mel", "mel_cqt"}:
            mel = compute_stereo_logmel_db(
                mixed,
                sr,
                n_fft=n_fft,
                hop=hop,
                win_length=win_length,
                n_mels=audio_cfg["n_mels"],
                fmin=audio_cfg["fmin"],
                fmax=audio_cfg["fmax"],
                window=window,
            )
            mel_tag = f"sr{sr}_dur{dur}_m{audio_cfg['n_mels']}_w{int(audio_cfg['win_ms'])}_{window}"
            mel_name = f"{mix_id}__{src_hash}__{mel_tag}.npy"
            mel_out = cache_root / "log_mels" / "mixes" / mel_name
            ensure_directory_exists(mel_out.parent)
            np.save(mel_out, mel.astype(np.float32))
            row["filepath"] = str(mel_out)

        if feature_type in {"cqt", "mel_cqt"}:
            cqt = compute_stereo_cqt_db(
                mixed,
                sr,
                n_bins=int(audio_cfg["n_bins"]),
                bins_per_octave=int(audio_cfg["bins_per_octave"]),
                hop_length=hop,
                fmin=float(audio_cfg["fmin"]),
            )
            cqt_tag = f"sr{sr}_dur{dur}_b{audio_cfg['n_bins']}_w{int(audio_cfg['win_ms'])}"
            cqt_name = f"{mix_id}__{src_hash}__{cqt_tag}.npy"
            cqt_out = cache_root / "log_cqt" / "mixes" / cqt_name
            ensure_directory_exists(cqt_out.parent)
            np.save(cqt_out, cqt.astype(np.float32))
            row["cqt_path"] = str(cqt_out)

        return True, row, None

    except Exception as exc:
        return False, {"mix_id": plan.get("mix_id", "unknown")}, str(exc)


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    ensure_directory_exists(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate synthetic mixtures and extract Mel/CQT features for multi-label training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config", default="src/configs/audio_params.yaml", help="Path to YAML config.")
    ap.add_argument(
        "--dataset",
        default="chinese_instruments",
        help="Dataset key under config['datasets'] used as the source pool.",
    )
    ap.add_argument(
        "--feature",
        default="mel_cqt",
        help="Feature family: mel | cqt | mel_cqt.",
    )
    ap.add_argument("--num_workers", type=int, default=8, help="Parallel worker processes.")
    ap.add_argument("--num_mixes", type=int, default=None, help="Override mixing.num_mixes from YAML.")
    ap.add_argument("--seed", type=int, default=None, help="Override mixing.seed from YAML.")
    ap.add_argument("--labels_config", default="src/configs/labels.yaml", help="Optional labels YAML for filtering.")
    ap.add_argument("--label_key", default=None, help="Optional key inside labels YAML (e.g. train_labels).")
    args = ap.parse_args()
    try:
        args.feature = normalize_feature_name(args.feature)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(2)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    datasets_cfg = cfg.get("datasets", {})
    if args.dataset not in datasets_cfg:
        print(f"Error: dataset '{args.dataset}' not found in {args.config}")
        sys.exit(1)

    audio_cfg = dict(cfg.get("audio", {}))
    if not audio_cfg:
        print(f"Error: missing 'audio' section in {args.config}")
        sys.exit(1)

    mixing_cfg = dict(cfg.get("mixing", {}))
    if args.num_mixes is not None:
        mixing_cfg["num_mixes"] = int(args.num_mixes)
    if args.seed is not None:
        mixing_cfg["seed"] = int(args.seed)

    num_mixes = int(mixing_cfg.get("num_mixes", 0))
    min_sources = int(mixing_cfg.get("min_sources", 2))
    max_sources = int(mixing_cfg.get("max_sources", 2))
    snr_db_min = float(mixing_cfg.get("snr_db_min", -3.0))
    snr_db_max = float(mixing_cfg.get("snr_db_max", 6.0))
    seed = int(mixing_cfg.get("seed", 1337))

    if num_mixes <= 0:
        print("Error: mixing.num_mixes must be > 0")
        sys.exit(1)
    if min_sources <= 0 or max_sources <= 0 or max_sources < min_sources:
        print("Error: invalid mixing min_sources/max_sources values")
        sys.exit(1)

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
            audio_cfg["n_bins"] = safe_n_bins

    dataset_cfg = datasets_cfg[args.dataset]
    train_dir = Path(dataset_cfg["train_dir"])
    cache_root_raw = dataset_cfg.get("mixed_cache_root")
    if not cache_root_raw:
        cache_root_raw = f"{dataset_cfg['cache_root']}_mixed"
    cache_root = Path(cache_root_raw)

    manifest_base = Path(dataset_cfg["manifest"]).resolve()
    manifest_stem = manifest_base.stem
    mel_manifest = manifest_base.with_name(f"{manifest_stem}_mels_mixed.csv")
    cqt_manifest = manifest_base.with_name(f"{manifest_stem}_cqt_mixed.csv")
    mel_cqt_manifest = manifest_base.with_name(f"{manifest_stem}_mel_cqt_mixed.csv")

    labels_config = Path(args.labels_config).expanduser() if args.labels_config else None
    allowed_labels = load_allowed_labels(labels_config, args.dataset, args.label_key)
    label_to_wavs = build_label_index(train_dir, allowed_labels)

    if not label_to_wavs:
        print(f"Error: no source WAV files found under {train_dir}")
        sys.exit(1)

    if len(label_to_wavs) < min_sources:
        print(
            f"Error: only {len(label_to_wavs)} labels with data found, "
            f"but mixing.min_sources={min_sources}"
        )
        sys.exit(1)

    if max_sources > len(label_to_wavs):
        print(
            f"[WARN] max_sources={max_sources} exceeds available labels={len(label_to_wavs)}. "
            f"Using max_sources={len(label_to_wavs)}."
        )
        max_sources = len(label_to_wavs)

    plans = build_mix_plan(
        label_to_wavs,
        num_mixes=num_mixes,
        min_sources=min_sources,
        max_sources=max_sources,
        snr_db_min=snr_db_min,
        snr_db_max=snr_db_max,
        seed=seed,
    )

    print(
        f"Generating {len(plans)} mixtures from dataset={args.dataset} "
        f"(labels={len(label_to_wavs)}, sources/mix={min_sources}-{max_sources}, "
        f"feature={args.feature})"
    )

    mel_rows: List[List[str]] = []
    cqt_rows: List[List[str]] = []
    mel_cqt_rows: List[List[str]] = []
    n_fail = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(_process_one_mix, plan, cache_root, audio_cfg, args.feature): plan["mix_id"]
            for plan in plans
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"MIX+DSP: {args.dataset}"):
            success, row, err = fut.result()
            if success:
                if "filepath" in row:
                    mel_rows.append([row["filepath"], row["labels"], row["sources"]])
                if "cqt_path" in row:
                    cqt_rows.append([row["cqt_path"], row["labels"], row["sources"]])
                if "filepath" in row and "cqt_path" in row:
                    mel_cqt_rows.append([row["filepath"], row["cqt_path"], row["labels"], row["sources"]])
            else:
                n_fail += 1
                print(f"\n[ERROR] {row.get('mix_id', 'unknown')}: {err}")

    if mel_rows:
        _write_csv(mel_manifest, ["filepath", "labels", "sources"], mel_rows)
        print(f"Saved mixed Mel manifest: {mel_manifest}")
    if cqt_rows:
        _write_csv(cqt_manifest, ["cqt_path", "labels", "sources"], cqt_rows)
        print(f"Saved mixed CQT manifest: {cqt_manifest}")
    if mel_cqt_rows:
        _write_csv(mel_cqt_manifest, ["filepath", "cqt_path", "labels", "sources"], mel_cqt_rows)
        print(f"Saved mixed Mel+CQT manifest: {mel_cqt_manifest}")

    n_success = len(plans) - n_fail
    print(f"Mix+extract complete. Success: {n_success}, Fail: {n_fail}")


if __name__ == "__main__":
    main()
