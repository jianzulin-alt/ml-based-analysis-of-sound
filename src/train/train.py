"""
Unified training entrypoint.

Features:
1) Interactive numeric selection for feature type when not provided.
2) Single script for Chinese/IRMAS and Mel/Mel+CQT training modes.
3) Reuses existing training loops to avoid rewriting core logic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def find_repo_root() -> Path:
    root = Path.cwd().resolve()
    while root != root.parent and not (root / "src").exists():
        root = root.parent
    return root


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _pick_feature_interactive() -> str:
    print("\nSelect feature mode for training:")
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


def _dataset_defaults(dataset: str, repo_root: Path) -> Dict[str, str | List[str]]:
    if dataset == "chinese":
        return {
            "manifests": ["data/processed/train_mels.csv"],
            "labels_yaml": "src/configs/labels.yaml",
            "audio_yaml": "src/configs/audio_params.yaml",
        }
    if dataset == "irmas":
        return {
            "manifests": ["data/processed/irmas_train_mels.csv"],
            "labels_yaml": "src/configs/labels_irmas.yaml",
            "audio_yaml": "src/configs/audio_params_irmas.yaml",
        }
    raise ValueError(f"Unsupported dataset: {dataset}")


def _default_run_name(dataset: str, feature: str) -> str:
    prefix = "Chinese" if dataset == "chinese" else "IRMAS"
    suffix = "mel" if feature == "mel" else "mel_cqt"
    return f"{prefix}_{suffix}_v1"


def _resolve_ckpt_dir(repo_root: Path, run_name: str, weights_dir: str | None) -> Path:
    if weights_dir:
        return Path(weights_dir).resolve()
    return (repo_root / "src" / "models" / "saved_weights" / run_name).resolve()


def _load_classes(repo_root: Path, labels_yaml: str) -> List[str]:
    labels_cfg = load_yaml(repo_root / labels_yaml)
    classes = [c.strip().lower() for c in labels_cfg.get("train_labels", [])]
    if not classes:
        raise ValueError(f"No train_labels found in {labels_yaml}")
    return classes


def _load_audio_cfg(repo_root: Path, audio_yaml: str) -> dict:
    audio_cfg_all = load_yaml(repo_root / audio_yaml)
    return audio_cfg_all.get("audio", audio_cfg_all)


def _build_run_cfg(
    *,
    dataset: str,
    feature: str,
    manifests: List[str],
    labels_yaml: str,
    audio_yaml: str,
    classes: List[str],
    args: argparse.Namespace,
) -> dict:
    return {
        "dataset": dataset,
        "feature_mode": feature,
        "manifests": manifests,
        "labels_yaml": labels_yaml,
        "audio_yaml": audio_yaml,
        "classes": classes,
        "train": {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs": args.epochs,
            "patience": args.patience,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "val_frac": args.val_frac,
            "seed": args.seed,
            "threshold": args.threshold,
            "num_workers": args.num_workers,
        },
        "resume": args.resume,
    }


def _train(
    *,
    feature: str,
    manifests: List[str],
    classes: List[str],
    ckpt_dir: Path,
    args: argparse.Namespace,
    audio_cfg: dict,
    resume_ckpt: Path | None,
) -> Tuple[object, dict]:
    from utils import multi_label_train_loop as train_loop_mel
    from utils_mel_cqt import multi_label_train_loop as train_loop_mel_cqt

    common_kwargs = dict(
        manifest_csv=manifests,
        classes=classes,
        ckpt_dir=ckpt_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_frac=args.val_frac,
        dropout=args.dropout,
        patience=args.patience,
        num_workers=args.num_workers,
        threshold=args.threshold,
        seed=args.seed,
        audio_cfg=audio_cfg,
        resume_from=resume_ckpt,
        save_best_stamped=False,
    )
    if feature == "mel":
        return train_loop_mel(**common_kwargs), common_kwargs
    if feature == "mel_cqt":
        return train_loop_mel_cqt(**common_kwargs), common_kwargs
    raise ValueError(f"Unsupported feature mode: {feature}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified train script")
    ap.add_argument("--dataset", choices=["chinese", "irmas"], default=None, help="Dataset preset")
    ap.add_argument("--feature", choices=["mel", "mel_cqt"], default=None, help="Feature mode")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--weights_dir", default=None)

    ap.add_argument("--manifests", nargs="+", default=None, help="Override manifest list")
    ap.add_argument("--labels_yaml", default=None)
    ap.add_argument("--audio_yaml", default=None)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    repo_root = find_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    print("Repo root:", repo_root)

    dataset = args.dataset or _pick_dataset_interactive()
    feature = args.feature or _pick_feature_interactive()
    run_name = args.run_name or _default_run_name(dataset, feature)

    ds_defaults = _dataset_defaults(dataset, repo_root)
    manifests = args.manifests or list(ds_defaults["manifests"])  # type: ignore[arg-type]
    labels_yaml = args.labels_yaml or str(ds_defaults["labels_yaml"])
    audio_yaml = args.audio_yaml or str(ds_defaults["audio_yaml"])

    classes = _load_classes(repo_root, labels_yaml)
    print(f"Loaded {len(classes)} classes: {', '.join(classes)}")

    audio_cfg = _load_audio_cfg(repo_root, audio_yaml)

    ckpt_dir = _resolve_ckpt_dir(repo_root, run_name, args.weights_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = _build_run_cfg(
        dataset=dataset,
        feature=feature,
        manifests=manifests,
        labels_yaml=labels_yaml,
        audio_yaml=audio_yaml,
        classes=classes,
        args=args,
    )
    with open(ckpt_dir / "run_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(run_cfg, f, sort_keys=False)

    resume_ckpt = ckpt_dir / "last.pt"
    if not args.resume or not resume_ckpt.exists():
        resume_ckpt = None
        print("Starting fresh (no resume).")
    else:
        print(f"Resuming from: {resume_ckpt}")

    print(f"Training mode: dataset={dataset}, feature={feature}, run={run_name}")
    results, _ = _train(
        feature=feature,
        manifests=manifests,
        classes=classes,
        ckpt_dir=ckpt_dir,
        args=args,
        audio_cfg=audio_cfg,
        resume_ckpt=resume_ckpt,
    )
    _ = results["history"]
    print("Training complete.")


if __name__ == "__main__":
    main()
