#!/usr/bin/env python3
"""
Generate synthetic multi-label mixed log-mel training features.

What this script does:
1. Reads source WAVs from a label-folder dataset layout: `<train_dir>/<label>/*.wav`
2. Randomly samples `k` labels per mix (`min_sources <= k <= max_sources`)
3. Mixes one source per label at random SNRs (`snr_db_min..snr_db_max`)
4. Converts each stereo mixture to log-mel spectrogram
5. Saves `.npy` features and writes a multilabel manifest (`labels` joined by `|`)

Defaults come from `audio_params.yaml`:
- `paths.train_dir`, `paths.mixed_cache_root`, `paths.mixed_manifest`, `paths.mixed_wav_debug_dir`
- `mixing.num_mixes`, `mixing.min_sources`, `mixing.max_sources`, `mixing.snr_db_min`,
  `mixing.snr_db_max`, `mixing.seed`, `mixing.save_wavs`, `mixing.max_wavs`

Any CLI argument passed in this script overrides the YAML value for that run.
"""

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

from src.preprocessing import (
    compute_stft_params,
    ensure_dir,
    ensure_duration,
    load_audio_as_stereo,
    stereo_to_logmel,
)


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def get_train_labels(cfg: dict) -> Optional[Set[str]]:
    labels = cfg.get("train_labels")
    if labels is None:
        labels = (cfg.get("dataset") or {}).get("train_labels")

    if not labels:
        return None

    out: Set[str] = set()
    for x in labels:
        if x is None:
            continue
        s = str(x).strip().lower()
        if s:
            out.add(s)
    return out or None


def iter_wavs_by_label(root: Path) -> Dict[str, List[Path]]:
    """
    Assumes folder structure: root/label/*.wav
    Returns mapping label -> wav paths.
    """
    by_label: Dict[str, List[Path]] = {}
    for wav in root.rglob("*.wav"):
        if wav.is_file():
            label = wav.parent.name.strip().lower()
            by_label.setdefault(label, []).append(wav)
    return by_label


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def mix_stereo_waveforms(
    stereos: List[np.ndarray],
    snr_db_range: Tuple[float, float],
) -> np.ndarray:
    """
    Mixes stereo waveforms (shape `(2, T)`).

    The first waveform is treated as base. Every additional waveform is scaled
    to a random SNR against that base, then added to the mixture.

    Returns peak-normalized stereo waveform `(2, T)`.
    """
    assert len(stereos) >= 2
    base = stereos[0].astype(np.float32, copy=True)
    base_rms = rms(base)

    mix = base
    for s in stereos[1:]:
        s = s.astype(np.float32, copy=False)
        s_rms = rms(s)

        snr_db = random.uniform(*snr_db_range)
        # scale so that 20*log10(base_rms / (gain*s_rms)) == snr_db
        gain = (base_rms / (s_rms + 1e-12)) * (10 ** (-snr_db / 20.0))
        mix = mix + s * float(gain)

    peak = float(np.max(np.abs(mix)) + 1e-12)
    mix = (0.99 * mix / peak).astype(np.float32)
    return mix


def save_mixed_npy(
    cache_root: Path,
    labels: List[str],
    idx: int,
    sr: int,
    dur: float,
    n_mels: int,
    win_ms: float,
    hop_ms: float,
    mel: np.ndarray,
) -> Path:
    """
    Saves mixed mel as:
      cache_root/<first_label>/[lab1_lab2...]mix_000001__sr..._dur..._m..._w..._h....npy
    """
    labels_norm = [l.strip().lower() for l in labels]
    labels_tag = "_".join(labels_norm)

    tag = f"sr{sr}_dur{dur}_m{n_mels}_w{int(win_ms)}_h{int(hop_ms)}"
    fn = f"[{labels_tag}]mix_{idx:06d}__{tag}.npy"

    out_dir = ensure_dir(cache_root / labels_norm[0])
    out_path = out_dir / fn
    np.save(out_path, mel.astype(np.float32))
    return out_path


def cfg_value(cli_value, cfg_section: dict, key: str, default):
    if cli_value is not None:
        return cli_value
    return cfg_section.get(key, default)


def bool_from_cfg(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Generate synthetic mixed-label log-mel spectrograms for training.\n"
            "Uses YAML config defaults and supports CLI overrides for each mix parameter."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument(
        "--config",
        default="configs/audio_params.yaml",
        help="Path to YAML config containing audio/paths/mixing sections.",
    )
    ap.add_argument(
        "--labels_file",
        help="Optional YAML containing train_labels allow-list. Overrides labels in --config.",
    )

    # Path overrides (fallback to config)
    ap.add_argument("--train_dir", default=None, help="Override paths.train_dir.")
    ap.add_argument("--out_cache_root", default=None, help="Override paths.mixed_cache_root.")
    ap.add_argument("--out_manifest", default=None, help="Override paths.mixed_manifest.")
    ap.add_argument("--wav_out_dir", default=None, help="Override paths.mixed_wav_debug_dir.")

    # Mixing parameter overrides (fallback to config)
    ap.add_argument("--num_mixes", type=int, default=None, help="Override mixing.num_mixes.")
    ap.add_argument("--min_sources", type=int, default=None, help="Override mixing.min_sources.")
    ap.add_argument("--max_sources", type=int, default=None, help="Override mixing.max_sources.")
    ap.add_argument("--snr_db_min", type=float, default=None, help="Override mixing.snr_db_min.")
    ap.add_argument("--snr_db_max", type=float, default=None, help="Override mixing.snr_db_max.")
    ap.add_argument("--seed", type=int, default=None, help="Override mixing.seed.")
    ap.add_argument("--max_wavs", type=int, default=None, help="Override mixing.max_wavs.")

    # Debug wav dump toggle with explicit on/off overrides
    ap.add_argument(
        "--save_wavs",
        dest="save_wavs",
        action="store_true",
        help="Force-enable saving debug mixed WAVs (overrides mixing.save_wavs).",
    )
    ap.add_argument(
        "--no_save_wavs",
        dest="save_wavs",
        action="store_false",
        help="Force-disable saving debug mixed WAVs (overrides mixing.save_wavs).",
    )
    ap.set_defaults(save_wavs=None)

    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    audio_cfg = cfg["audio"]
    path_cfg = cfg.get("paths", {})
    mix_cfg = cfg.get("mixing", {})

    sr = int(audio_cfg["sr"])
    dur = float(audio_cfg["duration"])
    n_mels = int(audio_cfg["n_mels"])
    win_ms = float(audio_cfg["win_ms"])
    hop_ms = float(audio_cfg["hop_ms"])
    fmin = float(audio_cfg["fmin"])
    fmax = audio_cfg.get("fmax")

    train_dir = Path(args.train_dir or path_cfg.get("train_dir", "data/train"))
    cache_root = Path(args.out_cache_root or path_cfg.get("mixed_cache_root", "data/processed/log_mels_mixed"))
    out_csv = Path(args.out_manifest or path_cfg.get("mixed_manifest", "data/processed/train_mels_mixed.csv"))
    wav_out_dir = Path(args.wav_out_dir or path_cfg.get("mixed_wav_debug_dir", "data/processed/debug/mixed_wavs"))

    num_mixes = int(cfg_value(args.num_mixes, mix_cfg, "num_mixes", 12000))
    min_sources = int(cfg_value(args.min_sources, mix_cfg, "min_sources", 2))
    max_sources = int(cfg_value(args.max_sources, mix_cfg, "max_sources", 2))
    snr_db_min = float(cfg_value(args.snr_db_min, mix_cfg, "snr_db_min", -3.0))
    snr_db_max = float(cfg_value(args.snr_db_max, mix_cfg, "snr_db_max", 6.0))
    seed = int(cfg_value(args.seed, mix_cfg, "seed", 1337))
    max_wavs = int(cfg_value(args.max_wavs, mix_cfg, "max_wavs", 50))

    if args.save_wavs is None:
        save_wavs = bool_from_cfg(mix_cfg.get("save_wavs"), default=False)
    else:
        save_wavs = args.save_wavs

    if num_mixes <= 0:
        raise ValueError(f"num_mixes must be > 0, got {num_mixes}")
    if min_sources < 2:
        raise ValueError(f"min_sources must be >= 2, got {min_sources}")
    if max_sources < min_sources:
        raise ValueError(f"max_sources ({max_sources}) must be >= min_sources ({min_sources})")
    if snr_db_min > snr_db_max:
        raise ValueError(f"snr_db_min ({snr_db_min}) must be <= snr_db_max ({snr_db_max})")
    if max_wavs < 0:
        raise ValueError(f"max_wavs must be >= 0, got {max_wavs}")
    if not train_dir.exists():
        raise FileNotFoundError(f"train_dir not found: {train_dir}")

    random.seed(seed)
    np.random.seed(seed)

    ensure_dir(cache_root)
    ensure_dir(out_csv.parent)
    if save_wavs:
        ensure_dir(wav_out_dir)

    allowed_labels = get_train_labels(cfg)
    if args.labels_file:
        labels_cfg = load_yaml(Path(args.labels_file))
        allowed_labels = get_train_labels(labels_cfg)

    by_label = iter_wavs_by_label(train_dir)
    labels = sorted(by_label.keys())

    if allowed_labels is not None:
        disk_labels = set(labels)
        missing = sorted(allowed_labels - disk_labels)
        extra = sorted(disk_labels - allowed_labels)
        if missing:
            print(f"WARNING: These train_labels were not found on disk: {missing}")
        if extra:
            print(f"INFO: These labels exist on disk but will be skipped: {extra}")

        labels = [lab for lab in labels if lab in allowed_labels]
        by_label = {lab: by_label[lab] for lab in labels}

    if len(labels) < min_sources:
        raise ValueError(f"Need at least {min_sources} labels to mix; found {len(labels)} in {train_dir}")
    if max_sources > len(labels):
        raise ValueError(f"max_sources={max_sources} exceeds available labels={len(labels)}")

    for lab in labels:
        if not by_label[lab]:
            raise ValueError(f"No wavs found for label '{lab}'")

    n_fft, hop, win_length = compute_stft_params(sr, win_ms, hop_ms)
    snr_range = (snr_db_min, snr_db_max)

    print("--- Generating Mixed Mels ---")
    print(f"Source: {train_dir}")
    print(f"Cache:  {cache_root}")
    print(f"Manifest: {out_csv}")
    print(f"Mixes: {num_mixes} | sources: {min_sources}-{max_sources} | snr_db: {snr_range}")
    print(f"Params: SR={sr}, Dur={dur}s, Mels={n_mels}, Win={win_ms}ms, Hop={hop_ms}ms")
    if save_wavs:
        print(f"Debug WAVs: saving first {max_wavs} mixes to {wav_out_dir}")

    rows_out: List[List[str]] = []

    for i in tqdm(range(num_mixes), desc="Mixing"):
        k = random.randint(min_sources, max_sources)
        chosen_labels = random.sample(labels, k)
        chosen_paths = [random.choice(by_label[lab]) for lab in chosen_labels]

        stereos = []
        for p in chosen_paths:
            stereo = load_audio_as_stereo(p, target_sr=sr)  # (2, T)
            stereo = ensure_duration(stereo, sr, dur)  # (2, target_T)
            stereos.append(stereo)

        mixed_stereo = mix_stereo_waveforms(stereos, snr_db_range=snr_range)

        if save_wavs and i < max_wavs:
            wav_path = wav_out_dir / f"mix_{i:06d}__[{'_'.join(chosen_labels)}]__sr{sr}.wav"
            sf.write(str(wav_path), mixed_stereo.T, sr, subtype="PCM_16")
            (wav_out_dir / f"mix_{i:06d}__[{'_'.join(chosen_labels)}].txt").write_text(
                "\n".join(chosen_labels) + "\n",
                encoding="utf-8",
            )

        mel = stereo_to_logmel(
            mixed_stereo,
            sr=sr,
            n_fft=n_fft,
            hop=hop,
            win_length=win_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )

        npy_path = save_mixed_npy(
            cache_root=cache_root,
            labels=chosen_labels,
            idx=i,
            sr=sr,
            dur=dur,
            n_mels=n_mels,
            win_ms=win_ms,
            hop_ms=hop_ms,
            mel=mel,
        )

        rows_out.append([npy_path.resolve().as_posix(), "|".join(chosen_labels)])

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "labels"])
        w.writerows(rows_out)

    print("Done.")
    print(f"Wrote {len(rows_out)} rows to: {out_csv}")


if __name__ == "__main__":
    main()
