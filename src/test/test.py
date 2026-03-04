"""
Unified testing entrypoint.

Features:
1) Interactive numeric selection for feature type when not provided.
2) Single script for Chinese/IRMAS and Mel/Mel+CQT test modes.
3) Reuses existing inference/evaluation utilities.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def find_repo_root() -> Path:
    root = Path.cwd().resolve()
    while root != root.parent and not (root / "src").exists():
        root = root.parent
    return root


def _pick_feature_interactive() -> str:
    print("\nSelect feature mode for testing:")
    print("  1. Mel")
    print("  2. Mel+CQT")
    raw = input("Enter number: ").strip()
    if raw == "1":
        return "mel"
    if raw == "2":
        return "mel_cqt"
    raise ValueError(f"Invalid selection: {raw}")


def _pick_dataset_interactive() -> str:
    print("\nSelect dataset preset:")
    print("  1. Chinese")
    print("  2. IRMAS")
    raw = input("Enter number: ").strip()
    if raw == "1":
        return "chinese"
    if raw == "2":
        return "irmas"
    raise ValueError(f"Invalid selection: {raw}")


def _default_weights_run(dataset: str, feature: str) -> str:
    prefix = "Chinese" if dataset == "chinese" else "IRMAS"
    suffix = "mel" if feature == "mel" else "mel_cqt"
    return f"{prefix}_{suffix}_v1"


def _default_manifest(dataset: str) -> str:
    if dataset == "chinese":
        return "data/test/a-touch-of-zen.csv"
    return "data/test/IRMAS/IRMAS-TestingData-Part1.csv"


def _resolve_path(repo_root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (repo_root / p).resolve()


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified test script")
    ap.add_argument("--dataset", choices=["chinese", "irmas"], default=None, help="Dataset preset")
    ap.add_argument("--feature", choices=["mel", "mel_cqt"], default=None, help="Feature mode")
    ap.add_argument("--weights_run", default=None)
    ap.add_argument("--weights_path", default=None, help="Override exact checkpoint path")
    ap.add_argument("--test_manifest", default=None, help="Override test manifest path")
    ap.add_argument("--threshold", type=float, default=None, help="If omitted, auto-tunes threshold")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no_progress", action="store_true")
    ap.add_argument("--cqt_bins", type=int, default=None)
    ap.add_argument("--bins_per_octave", type=int, default=12)
    args = ap.parse_args()

    repo_root = find_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    print("Repo root:", repo_root)

    from src.models.CRNN import CRNN
    from src.test.utils import (
        evaluate_multilabel_performance as evaluate_mel,
        find_best_threshold as find_best_threshold_mel,
        run_inference as run_inference_mel,
    )
    from src.test.utils_mel_cqt import (
        run_inference as run_inference_mel_cqt,
    )

    dataset = args.dataset or _pick_dataset_interactive()
    feature = args.feature or _pick_feature_interactive()

    weights_run = args.weights_run or _default_weights_run(dataset, feature)
    test_manifest = args.test_manifest or _default_manifest(dataset)

    if args.weights_path:
        weights_path = _resolve_path(repo_root, args.weights_path)
    else:
        weights_path = _resolve_path(
            repo_root, f"src/models/saved_weights/{weights_run}/best_val.pt"
        )

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    test_manifest_path = _resolve_path(repo_root, test_manifest)
    if not test_manifest_path.exists():
        raise FileNotFoundError(f"Test manifest not found: {test_manifest_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Test mode: dataset={dataset}, feature={feature}, weights_run={weights_run}")

    if feature == "mel":
        preds_arr, gts_arr, sample_ids, _, valid_labels, _ = run_inference_mel(
            model_cls=CRNN,
            model_kwargs={"in_ch": 2},
            model_weights_path=weights_path,
            device=device,
            test_manifest_csv=test_manifest_path,
            root=repo_root,
            show_progress=not args.no_progress,
        )
    else:
        preds_arr, gts_arr, sample_ids, _, valid_labels, _ = run_inference_mel_cqt(
            model_cls=CRNN,
            model_kwargs={"in_ch": 4},
            model_weights_path=weights_path,
            device=device,
            test_manifest_csv=test_manifest_path,
            root=repo_root,
            show_progress=not args.no_progress,
            cqt_bins=args.cqt_bins,
            bins_per_octave=args.bins_per_octave,
        )

    if args.threshold is None:
        best_t = find_best_threshold_mel(preds_arr, gts_arr, valid_labels)
        print(f"Best threshold found: {best_t:.2f}")
        threshold = best_t
    else:
        threshold = args.threshold

    _ = evaluate_mel(
        all_preds=preds_arr,
        all_gt=gts_arr,
        class_list=valid_labels,
        sample_ids=sample_ids,
        threshold=threshold,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
