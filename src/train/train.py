"""
Unified training entrypoint.

Features:
1) Interactive numeric selection for dataset/feature/task mode when omitted.
2) Supports Mel / CQT / Mel+CQT feature modes.
3) Supports single-label and multi-label training modes.
4) In multi-label mode, auto-checks mixed-data manifest and can trigger generation.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def find_repo_root() -> Path:
    root = Path.cwd().resolve()
    while root != root.parent and not (root / "src").exists():
        root = root.parent
    return root


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _read_csv_with_fallback(path: Path) -> pd.DataFrame:
    encodings = ("utf-8", "utf-8-sig", "gbk", "cp936")
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def _pick_feature_interactive() -> str:
    print("\nSelect feature mode for training:")
    print("  1. Mel")
    print("  2. CQT")
    print("  3. Mel+CQT")
    raw = input("Enter number: ").strip()
    if raw == "1":
        return "mel"
    if raw == "2":
        return "cqt"
    if raw == "3":
        return "mel_cqt"
    raise ValueError(f"Invalid selection: {raw}")


def _pick_task_mode_interactive() -> str:
    print("\nSelect task mode:")
    print("  1. Single-label")
    print("  2. Multi-label")
    raw = input("Enter number: ").strip()
    if raw == "1":
        return "single_label"
    if raw == "2":
        return "multi_label"
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


def _dataset_defaults(dataset: str) -> Dict[str, str | List[str]]:
    if dataset == "chinese":
        return {
            "manifests": ["data/processed/train_mels.csv"],
            "mixed_manifest": "data/processed/train_mels_mixed.csv",
            "labels_yaml": "src/configs/labels.yaml",
            "audio_yaml": "src/configs/audio_params.yaml",
        }
    if dataset == "irmas":
        return {
            "manifests": ["data/processed/irmas_train_mels.csv"],
            "mixed_manifest": "",
            "labels_yaml": "src/configs/labels_irmas.yaml",
            "audio_yaml": "src/configs/audio_params_irmas.yaml",
        }
    raise ValueError(f"Unsupported dataset: {dataset}")


def _default_run_name(dataset: str, feature: str, task_mode: str) -> str:
    prefix = "Chinese" if dataset == "chinese" else "IRMAS"
    return f"{prefix}_{feature}_{task_mode}_v1"


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
    task_mode: str,
    manifests: List[str],
    labels_yaml: str,
    audio_yaml: str,
    classes: List[str],
    args: argparse.Namespace,
) -> dict:
    return {
        "dataset": dataset,
        "feature_mode": feature,
        "task_mode": task_mode,
        "backbone": args.backbone,
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
        "mixed_data_policy": {
            "auto_generate_mixed": args.auto_generate_mixed,
            "num_mixes": args.num_mixes,
        },
        "resume": args.resume,
    }


def _ask_yes_no(prompt: str, default_yes: bool = True) -> bool:
    suffix = " [Y/n]: " if default_yes else " [y/N]: "
    try:
        raw = input(prompt + suffix).strip().lower()
    except EOFError:
        # Notebook/CI subprocess may not have interactive stdin.
        print(f"[mix-check] Non-interactive stdin detected; using default={default_yes}.")
        return default_yes
    if raw == "":
        return default_yes
    return raw in {"y", "yes"}


def _mixed_manifest_ready(manifest_path: Path, feature: str) -> Tuple[bool, str]:
    if not manifest_path.exists():
        return False, f"missing file: {manifest_path}"

    try:
        df = _read_csv_with_fallback(manifest_path)
    except Exception as exc:
        return False, f"cannot read manifest ({exc})"

    if df.empty:
        return False, "manifest has 0 rows"
    if "labels" not in df.columns:
        return False, "manifest lacks 'labels' column"

    labels_ser = df["labels"].fillna("").astype(str)
    has_multi = labels_ser.str.contains(r"[|,;]").any()
    if not bool(has_multi):
        return False, "manifest has no multi-label rows"

    if feature in {"cqt", "mel_cqt"}:
        if "cqt_path" not in df.columns:
            return False, "manifest lacks 'cqt_path' column"
        cqt_ok = df["cqt_path"].fillna("").astype(str).str.strip().ne("").any()
        if not bool(cqt_ok):
            return False, "manifest has empty 'cqt_path'"

    return True, "ok"


def _ensure_manifest_columns(manifest_paths: List[str], feature: str) -> None:
    need_cqt = feature in {"cqt", "mel_cqt"}
    for m in manifest_paths:
        p = Path(m)
        if not p.exists():
            raise FileNotFoundError(f"Manifest not found: {p}")
        df = _read_csv_with_fallback(p)
        if need_cqt and "cqt_path" not in df.columns:
            raise ValueError(
                f"Manifest missing 'cqt_path': {p}. "
                "Generate CQT features first (e.g. `make all` or CQT generate task)."
            )


def _run_generate_task(repo_root: Path, task: str, num_workers: int, num_mixes: int | None) -> None:
    env = os.environ.copy()
    src_path = str((repo_root / "src").resolve())
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path if not current else f"{src_path}{os.pathsep}{current}"
    env["NUM_WORKERS"] = str(max(1, num_workers))
    if num_mixes is not None:
        env["NUM_MIXES"] = str(max(1, int(num_mixes)))

    cmd = [sys.executable, "src/scripts/generate.py", "--task", task]
    print("[mix-check] Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=repo_root, env=env, check=True)


def _feature_display_name(feature: str) -> str:
    fmap = {
        "mel": "Mel",
        "cqt": "CQT",
        "mel_cqt": "Mel + CQT",
    }
    return fmap.get(str(feature).strip().lower(), str(feature))


def _last_or_nan(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(values[-1])


def _print_final_metrics(history: dict, task_mode: str) -> None:
    train_loss = _last_or_nan(history.get("train_loss", []))
    val_loss = _last_or_nan(history.get("val_loss", []))
    train_acc = _last_or_nan(history.get("train_acc", []))
    val_acc = _last_or_nan(history.get("val_acc", []))
    train_macro = _last_or_nan(history.get("train_macro_f1", []))
    val_macro = _last_or_nan(history.get("val_macro_f1", []))

    print("\nFinal Metrics")
    print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"  Train Acc:  {train_acc:.4f} | Val Acc:  {val_acc:.4f}")
    if task_mode == "multi_label":
        train_micro = _last_or_nan(history.get("train_micro_f1", []))
        val_micro = _last_or_nan(history.get("val_micro_f1", []))
        print(f"  Train MicroF1: {train_micro:.4f} | Val MicroF1: {val_micro:.4f}")
    print(f"  Train MacroF1: {train_macro:.4f} | Val MacroF1: {val_macro:.4f}")


def _save_training_plots(
    *,
    history: dict,
    feature: str,
    task_mode: str,
    run_name: str,
    ckpt_dir: Path,
) -> Tuple[Path, Path]:
    if not history.get("train_loss"):
        raise ValueError("History is empty; cannot plot training curves.")

    def _plot(ax, key: str, label: str) -> None:
        vals = history.get(key, [])
        if not vals:
            return
        x = range(1, len(vals) + 1)
        ax.plot(x, vals, label=label)

    feature_name = _feature_display_name(feature)
    title_prefix = f"{feature_name} | {task_mode} | {run_name}"

    fig = plt.figure(figsize=(18, 5))

    ax1 = fig.add_subplot(1, 3, 1)
    _plot(ax1, "train_loss", "Train Loss")
    _plot(ax1, "val_loss", "Val Loss")
    ax1.set_title(f"{feature_name} Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    if ax1.lines:
        ax1.legend()

    ax2 = fig.add_subplot(1, 3, 2)
    _plot(ax2, "train_acc", "Train Acc")
    _plot(ax2, "val_acc", "Val Acc")
    ax2.set_title(f"{feature_name} Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, alpha=0.3)
    if ax2.lines:
        ax2.legend()

    ax3 = fig.add_subplot(1, 3, 3)
    if history.get("train_micro_f1") and history.get("val_micro_f1"):
        _plot(ax3, "train_micro_f1", "Train Micro F1")
        _plot(ax3, "val_micro_f1", "Val Micro F1")
    _plot(ax3, "train_macro_f1", "Train Macro F1")
    _plot(ax3, "val_macro_f1", "Val Macro F1")
    ax3.set_title(f"{feature_name} F1")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("F1")
    ax3.grid(True, alpha=0.3)
    if ax3.lines:
        ax3.legend()

    fig.suptitle(f"Training Curves: {title_prefix}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    png_path = ckpt_dir / f"training_curves_{feature}_{task_mode}.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    history_df = pd.DataFrame(history)
    csv_path = ckpt_dir / f"training_history_{feature}_{task_mode}.csv"
    history_df.to_csv(csv_path, index=False, encoding="utf-8")
    return png_path, csv_path


def _resolve_manifests(
    *,
    repo_root: Path,
    dataset: str,
    feature: str,
    task_mode: str,
    args: argparse.Namespace,
    defaults: Dict[str, str | List[str]],
) -> List[str]:
    if args.manifests:
        manifests = [str(Path(m).resolve()) for m in args.manifests]
        _ensure_manifest_columns(manifests, feature)
        return manifests

    manifests = [str((repo_root / p).resolve()) for p in list(defaults["manifests"])]  # type: ignore[arg-type]
    if task_mode != "multi_label":
        _ensure_manifest_columns(manifests, feature)
        return manifests

    # Multi-label mode: for Chinese dataset, prefer mixed manifest and auto-check.
    if dataset != "chinese":
        print("[mix-check] Multi-label mode enabled. Auto mixed-data check currently applies to Chinese dataset only.")
        _ensure_manifest_columns(manifests, feature)
        return manifests

    mixed_rel = str(defaults.get("mixed_manifest") or "")
    if not mixed_rel:
        _ensure_manifest_columns(manifests, feature)
        return manifests

    mixed_manifest = (repo_root / mixed_rel).resolve()
    ok, reason = _mixed_manifest_ready(mixed_manifest, feature)
    if ok:
        print(f"[mix-check] Using mixed manifest: {mixed_manifest}")
        return [str(mixed_manifest)]

    print(f"[mix-check] Mixed manifest not ready ({reason}).")
    task = "mixed_mel" if feature == "mel" else "mixed_mel_cqt"

    should_generate = False
    if args.auto_generate_mixed == "always":
        should_generate = True
    elif args.auto_generate_mixed == "ask":
        should_generate = _ask_yes_no(
            f"Generate mixed data now with task '{task}'?",
            default_yes=True,
        )
    elif args.auto_generate_mixed == "never":
        should_generate = False

    if should_generate:
        _run_generate_task(repo_root, task, args.num_workers, args.num_mixes)
        ok2, reason2 = _mixed_manifest_ready(mixed_manifest, feature)
        if not ok2:
            raise RuntimeError(f"Mixed data generation finished but manifest still invalid: {reason2}")
        print(f"[mix-check] Mixed data ready: {mixed_manifest}")
        return [str(mixed_manifest)]

    print("[mix-check] Continue without mixed manifest (multi-label on non-mixed data).")
    _ensure_manifest_columns(manifests, feature)
    return manifests


def _build_dataset_for_single(
    *,
    feature: str,
    manifests: List[str],
    classes: List[str],
    repo_root: Path,
):
    from src.data_loader import MultiLabelMelDataset
    from src.data_loader_cqt import MultiLabelCqtDataset
    from src.data_loader_mel_cqt import MultiLabelMelCqtDataset

    kwargs = dict(manifest_csv=manifests, class_names=classes, project_root=repo_root)
    if feature == "mel":
        return MultiLabelMelDataset(**kwargs)
    if feature == "cqt":
        return MultiLabelCqtDataset(**kwargs)
    if feature == "mel_cqt":
        return MultiLabelMelCqtDataset(**kwargs)
    raise ValueError(f"Unsupported feature mode: {feature}")


def _train(
    *,
    feature: str,
    task_mode: str,
    manifests: List[str],
    classes: List[str],
    ckpt_dir: Path,
    args: argparse.Namespace,
    audio_cfg: dict,
    resume_ckpt: Path | None,
    repo_root: Path,
    backbone: str,
) -> Tuple[object, dict]:
    if task_mode == "multi_label":
        from utils import multi_label_train_loop as train_loop_mel
        from utils_mel_cqt import multi_label_train_loop as train_loop_cqt_or_melcqt

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
            backbone=backbone,
        )
        if feature == "mel":
            return train_loop_mel(**common_kwargs), common_kwargs
        if feature in {"cqt", "mel_cqt"}:
            return train_loop_cqt_or_melcqt(
                **common_kwargs,
                feature_mode=feature,
                in_ch=None,
            ), common_kwargs
        raise ValueError(f"Unsupported feature mode: {feature}")

    if task_mode == "single_label":
        from utils_single import single_label_train_loop

        dataset = _build_dataset_for_single(
            feature=feature,
            manifests=manifests,
            classes=classes,
            repo_root=repo_root,
        )
        kwargs = dict(
            dataset=dataset,
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
            seed=args.seed,
            audio_cfg=audio_cfg,
            feature_mode=feature,
            resume_from=resume_ckpt,
            backbone=backbone,
        )
        return single_label_train_loop(**kwargs), kwargs

    raise ValueError(f"Unsupported task mode: {task_mode}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified train script")
    ap.add_argument("--dataset", choices=["chinese", "irmas"], default=None, help="Dataset preset")
    ap.add_argument("--feature", choices=["mel", "cqt", "mel_cqt"], default=None, help="Feature mode")
    ap.add_argument(
        "--task_mode",
        choices=["single_label", "multi_label"],
        default=None,
        help="Training target mode",
    )
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--weights_dir", default=None)

    ap.add_argument("--manifests", nargs="+", default=None, help="Override manifest list")
    ap.add_argument("--labels_yaml", default=None)
    ap.add_argument("--audio_yaml", default=None)

    ap.add_argument(
        "--auto_generate_mixed",
        choices=["ask", "always", "never"],
        default="ask",
        help="Only used for multi-label training on Chinese dataset.",
    )
    ap.add_argument("--num_mixes", type=int, default=None, help="Optional override for mixed generation count.")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument(
        "--backbone",
        choices=["crnn", "densenet121"],
        default="densenet121",
        help="Model backbone used by training and written into checkpoint metadata.",
    )
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--threshold", type=float, default=0.5, help="Used only for multi-label mode.")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    repo_root = find_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    print("Repo root:", repo_root)

    dataset = args.dataset or _pick_dataset_interactive()
    feature = args.feature or _pick_feature_interactive()
    task_mode = args.task_mode or _pick_task_mode_interactive()
    run_name = args.run_name or _default_run_name(dataset, feature, task_mode)

    ds_defaults = _dataset_defaults(dataset)
    labels_yaml = args.labels_yaml or str(ds_defaults["labels_yaml"])
    audio_yaml = args.audio_yaml or str(ds_defaults["audio_yaml"])
    manifests = _resolve_manifests(
        repo_root=repo_root,
        dataset=dataset,
        feature=feature,
        task_mode=task_mode,
        args=args,
        defaults=ds_defaults,
    )

    classes = _load_classes(repo_root, labels_yaml)
    print(f"Loaded {len(classes)} classes: {', '.join(classes)}")

    audio_cfg = _load_audio_cfg(repo_root, audio_yaml)

    ckpt_dir = _resolve_ckpt_dir(repo_root, run_name, args.weights_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = _build_run_cfg(
        dataset=dataset,
        feature=feature,
        task_mode=task_mode,
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

    print(
        f"Training mode: dataset={dataset}, feature={feature}, task_mode={task_mode}, "
        f"backbone={args.backbone}, run={run_name}"
    )
    results, _ = _train(
        feature=feature,
        task_mode=task_mode,
        manifests=manifests,
        classes=classes,
        ckpt_dir=ckpt_dir,
        args=args,
        audio_cfg=audio_cfg,
        resume_ckpt=resume_ckpt,
        repo_root=repo_root,
        backbone=args.backbone,
    )
    history = results["history"]
    _print_final_metrics(history, task_mode)
    try:
        plot_path, history_csv = _save_training_plots(
            history=history,
            feature=feature,
            task_mode=task_mode,
            run_name=run_name,
            ckpt_dir=ckpt_dir,
        )
        print(f"Saved training curves: {plot_path}")
        print(f"Saved training history: {history_csv}")
    except Exception as exc:
        print(f"[WARN] Failed to save training plots/history: {exc}")
    print("Training complete.")


if __name__ == "__main__":
    main()
