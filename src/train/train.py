"""Usage examples
python -m src.train.train \
  --run_name CNN_v2 \
  --model cnn \
  --manifests data/processed/train_mels.csv data/processed/train_mels_mixed.csv \
  --labels_yaml src/configs/labels.yaml \
  --audio_yaml src/configs/audio_params.yaml

python -m src.train.train \
  --run_name MobileNetV3_v1 \
  --model mobilenet_v3_small \
  --pretrained \
  --freeze_backbone_epochs 8 \
  --manifests data/processed/train_mels.csv data/processed/train_mels_mixed.csv \
  --labels_yaml src/configs/labels.yaml \
  --audio_yaml src/configs/audio_params.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import src
import yaml

from src.train.train_multilabel import multi_label_train_loop
from src.train.utils import SUPPORTED_MODELS

REPO_ROOT = Path(src.__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (REPO_ROOT / p).resolve()


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="CNN_v1")
    ap.add_argument(
        "--weights_dir",
        default=None,
        help="Override weights dir; default uses src/models/saved_weights/<run_name>",
    )
    ap.add_argument("--manifests", nargs="+", required=True, help="One or more manifest CSVs")
    ap.add_argument("--labels_yaml", required=True)
    ap.add_argument("--audio_yaml", required=True)

    ap.add_argument("--model", type=str, default="cnn", choices=SUPPORTED_MODELS)
    ap.add_argument("--in_ch", type=int, default=2)
    ap.add_argument("--pretrained", action="store_true", help="Use pretrained weights when supported.")
    ap.add_argument(
        "--freeze_backbone_epochs",
        type=int,
        default=0,
        help="Freeze model feature extractor for N epochs (supported models only).",
    )

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--resume", action="store_true", help="Resume from last.pt if present")
    ap.add_argument("--warm_start", action="store_true", help="Load weights only, no optimizer/scheduler state.")
    ap.add_argument("--warm_start_from", default=None, help="Path to checkpoint for warm start.")
    ap.add_argument(
        "--experiment_log",
        default="src/models/experiments.csv",
        help="CSV to append run summary row. Pass empty string to disable.",
    )
    args = ap.parse_args()

    print("Repo root:", REPO_ROOT)

    labels_yaml = resolve_path(args.labels_yaml)
    audio_yaml = resolve_path(args.audio_yaml)
    manifests = [str(resolve_path(m)) for m in args.manifests]

    labels_cfg = load_yaml(labels_yaml)
    classes = [c.strip().lower() for c in labels_cfg.get("train_labels", [])]
    if not classes:
        raise ValueError(f"No train_labels found in {labels_yaml}")
    print(f"Loaded {len(classes)} classes: {', '.join(classes)}")

    audio_cfg_all = load_yaml(audio_yaml)
    audio_cfg = audio_cfg_all.get("audio", audio_cfg_all)

    if args.weights_dir:
        ckpt_dir = resolve_path(args.weights_dir)
    else:
        ckpt_dir = (REPO_ROOT / "src" / "models" / "saved_weights" / args.run_name).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    experiment_log = None
    if args.experiment_log.strip():
        experiment_log = resolve_path(args.experiment_log)

    resume_ckpt = None
    if args.resume:
        candidate = ckpt_dir / "last.pt"
        if candidate.exists():
            resume_ckpt = candidate
            print(f"Resuming from: {resume_ckpt}")
        else:
            print("Resume requested, but last.pt not found. Starting fresh.")

    warm_start_ckpt = None
    if (args.warm_start or args.warm_start_from) and resume_ckpt is None:
        candidate = resolve_path(args.warm_start_from) if args.warm_start_from else (ckpt_dir / "last.pt")
        if candidate.exists():
            warm_start_ckpt = candidate
            print(f"Warm-starting model weights from: {warm_start_ckpt}")
        else:
            print(f"Warm-start checkpoint not found at {candidate}. Starting fresh.")
    elif (args.warm_start or args.warm_start_from) and resume_ckpt is not None:
        print("Ignoring warm-start because --resume was provided.")

    run_cfg = {
        "run_name": args.run_name,
        "model": {
            "name": args.model,
            "in_ch": args.in_ch,
            "pretrained": args.pretrained,
            "freeze_backbone_epochs": args.freeze_backbone_epochs,
        },
        "manifests": manifests,
        "labels_yaml": str(labels_yaml),
        "audio_yaml": str(audio_yaml),
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
        "resume_from": str(resume_ckpt) if resume_ckpt else None,
        "warm_start_from": str(warm_start_ckpt) if warm_start_ckpt else None,
        "experiment_log": str(experiment_log) if experiment_log else None,
    }
    with open(ckpt_dir / "run_config.yaml", "w") as f:
        yaml.safe_dump(run_cfg, f, sort_keys=False)

    results = multi_label_train_loop(
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
        model_name=args.model,
        in_ch=args.in_ch,
        pretrained=args.pretrained,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        warm_start_from=warm_start_ckpt,
        run_name=args.run_name,
        experiment_log=experiment_log,
    )

    print("Training complete.")
    print("Run summary:")
    for key, value in results["summary"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
