from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.inference.utils import PROJECT_ROOT, run_inference_on_wav_files
from src.models import CNN

DEFAULT_WAV_PATHS = [
    "data/test/other/Come Drink with Me - Drunken Cat Song.wav",
    "data/test/other/a_touch_of_zen_music.wav",
]
DEFAULT_WEIGHTS_PATH = "src/models/saved_weights/CNN_v1/best_val.pt"


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run model inference directly on one or more WAV files."
    )
    parser.add_argument(
        "--wav",
        nargs="+",
        default=DEFAULT_WAV_PATHS,
        help="WAV paths (relative to project root or absolute).",
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS_PATH,
        help="Path to checkpoint .pt file.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Torch device selection.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold used to list detected labels.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many highest-probability labels to print per file.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = select_device(args.device)
    weights_path = Path(args.weights)
    resolved_weights = (
        weights_path if weights_path.is_absolute() else (PROJECT_ROOT / weights_path)
    ).resolve()
    top_k = max(1, int(args.top_k))

    preds_arr, sample_ids, resolved_wav_paths, _, valid_labels = run_inference_on_wav_files(
        model_cls=CNN,
        model_kwargs={"in_ch": 2},
        model_weights_path=resolved_weights,
        device=device,
        wav_paths=args.wav,
        root=PROJECT_ROOT,
        show_progress=not args.no_progress,
    )

    print(f"\nDevice: {device}")
    print(f"Weights: {resolved_weights}")
    print(f"Threshold: {args.threshold:.2f}")

    for wav_path, sample_id, probs in zip(resolved_wav_paths, sample_ids, preds_arr):
        print("\n" + "=" * 80)
        print(f"Sample: {sample_id}")
        print(f"File:   {wav_path}")

        top_indices = np.argsort(probs)[::-1][: min(top_k, len(valid_labels))]
        print(f"Top-{len(top_indices)} predictions:")
        for idx in top_indices:
            print(f"  {valid_labels[idx]:<15} {probs[idx]:.4f}")

        above_threshold = [
            (valid_labels[i], probs[i])
            for i in np.argsort(probs)[::-1]
            if probs[i] >= args.threshold
        ]
        if above_threshold:
            formatted = ", ".join(f"{label} ({score:.3f})" for label, score in above_threshold)
            print(f"Detected labels (>= {args.threshold:.2f}): {formatted}")
        else:
            print(f"Detected labels (>= {args.threshold:.2f}): none")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
