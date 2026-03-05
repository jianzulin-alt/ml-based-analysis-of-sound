"""
Unified testing entrypoint.

Features:
1) Interactive numeric selection for feature mode when omitted.
2) Supports Mel / CQT / Mel+CQT.
3) Supports multi-label and single-label evaluation modes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def find_repo_root() -> Path:
    root = Path.cwd().resolve()
    while root != root.parent and not (root / "src").exists():
        root = root.parent
    return root


def _pick_feature_interactive() -> str:
    print("\nSelect feature mode for testing:")
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


def _default_weights_run(dataset: str, feature: str, task_mode: str) -> str:
    prefix = "Chinese" if dataset == "chinese" else "IRMAS"
    return f"{prefix}_{feature}_{task_mode}_v1"


def _default_manifest(dataset: str) -> str:
    if dataset == "chinese":
        return "data/test/a-touch-of-zen.csv"
    return "data/test/IRMAS/IRMAS-TestingData-Part1.csv"


def _resolve_path(repo_root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (repo_root / p).resolve()


def _feature_display_name(feature: str) -> str:
    fmap = {
        "mel": "Mel",
        "cqt": "CQT",
        "mel_cqt": "Mel + CQT",
    }
    return fmap.get(str(feature).strip().lower(), str(feature))


def _infer_task_mode_from_ckpt(weights_path: Path) -> str:
    ckpt = torch.load(weights_path, map_location="cpu")
    return str(ckpt.get("task_mode", "multi_label")).strip().lower()


def _infer_backbone_from_ckpt(weights_path: Path) -> str:
    ckpt = torch.load(weights_path, map_location="cpu")
    return str(ckpt.get("backbone", "crnn")).strip().lower()


def _resolve_model_cls(backbone: str):
    name = str(backbone).strip().lower()
    if name == "crnn":
        from src.models.CRNN import CRNN

        return CRNN
    if name in {"densenet121", "densenet_121", "cnn_densenet_121"}:
        from src.models.CNN_DenseNet_121 import CNN_DenseNet_121

        return CNN_DenseNet_121
    raise ValueError(f"Unsupported backbone: {backbone}")


def _auto_pick_weights_path(repo_root: Path, dataset: str, feature: str) -> Path:
    """
    Try both task-mode default run names and pick the first existing checkpoint.
    Preference order: multi_label, single_label.
    """
    for tm in ("multi_label", "single_label"):
        run = _default_weights_run(dataset, feature, tm)
        p = _resolve_path(repo_root, f"src/models/saved_weights/{run}/best_val.pt")
        if p.exists():
            return p
    # Fallback path (keeps old behavior shape for error message)
    fallback = _default_weights_run(dataset, feature, "multi_label")
    return _resolve_path(repo_root, f"src/models/saved_weights/{fallback}/best_val.pt")


def _evaluate_single_label(preds_arr: np.ndarray, gts_arr: np.ndarray, valid_labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if preds_arr.size == 0 or gts_arr.size == 0:
        raise RuntimeError("No valid samples available for evaluation.")

    gt_counts = gts_arr.sum(axis=1)
    valid_mask = gt_counts > 0
    if not bool(valid_mask.any()):
        raise RuntimeError("No samples with non-empty ground truth labels.")

    if bool((gt_counts[valid_mask] > 1).any()):
        multi_rows = int((gt_counts[valid_mask] > 1).sum())
        print(f"[WARN] Found {multi_rows} samples with multi-positive GT in single-label eval; using argmax target.")

    y_true = np.argmax(gts_arr[valid_mask], axis=1)
    y_pred = np.argmax(preds_arr[valid_mask], axis=1)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"Single-label accuracy: {acc:.4f}")
    print(f"Single-label macro F1: {macro_f1:.4f}")

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(valid_labels))),
        target_names=valid_labels,
        zero_division=0,
    )
    print("\n--- Classification Report ---")
    print(report)
    return y_true, y_pred


def _save_single_label_eval_plot(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    feature: str,
    ckpt_path: Path,
) -> Path:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(f"Confusion Matrix - {_feature_display_name(feature)}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Limit tick labels for readability when class count is large.
    if len(labels) <= 30:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)

    fig.tight_layout()
    out_path = ckpt_path.parent / f"test_eval_single_{feature}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _save_multilabel_eval_plot(
    *,
    preds_arr: np.ndarray,
    gts_arr: np.ndarray,
    feature: str,
    ckpt_path: Path,
) -> Path:
    if preds_arr.size == 0 or gts_arr.size == 0:
        raise RuntimeError("No valid samples available for evaluation plot.")

    thresholds = np.arange(0.05, 1.00, 0.05)
    micro_f1_list = []
    macro_f1_list = []
    subset_acc_list = []

    gts = gts_arr.astype(int)
    for t in thresholds:
        preds_bin = (preds_arr >= t).astype(int)
        micro_f1_list.append(f1_score(gts, preds_bin, average="micro", zero_division=0))
        macro_f1_list.append(f1_score(gts, preds_bin, average="macro", zero_division=0))
        subset_acc_list.append(accuracy_score(gts, preds_bin))

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(thresholds, micro_f1_list, marker="o", label="Micro F1")
    ax.plot(thresholds, macro_f1_list, marker="s", label="Macro F1")
    ax.plot(thresholds, subset_acc_list, marker="^", label="Subset Acc")
    ax.set_title(f"Threshold Curves - {_feature_display_name(feature)}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_path = ckpt_path.parent / f"test_eval_multilabel_{feature}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified test script")
    ap.add_argument("--dataset", choices=["chinese", "irmas"], default=None, help="Dataset preset")
    ap.add_argument("--feature", choices=["mel", "cqt", "mel_cqt"], default=None, help="Feature mode")
    ap.add_argument("--task_mode", choices=["single_label", "multi_label"], default=None, help="Eval mode")
    ap.add_argument("--backbone", choices=["crnn", "densenet121"], default=None, help="Model backbone.")
    ap.add_argument("--weights_run", default=None)
    ap.add_argument("--weights_path", default=None, help="Override exact checkpoint path")
    ap.add_argument("--test_manifest", default=None, help="Override test manifest path")
    ap.add_argument("--threshold", type=float, default=None, help="Only for multi-label mode; None -> auto tune")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no_progress", action="store_true")
    ap.add_argument("--cqt_bins", type=int, default=None)
    ap.add_argument("--bins_per_octave", type=int, default=12)
    args = ap.parse_args()

    repo_root = find_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    print("Repo root:", repo_root)

    from src.test.utils import (
        evaluate_multilabel_performance as evaluate_mel,
        find_best_threshold as find_best_threshold_mel,
        run_inference as run_inference_mel,
    )
    from src.test.utils_mel_cqt import (
        run_inference as run_inference_mel_cqt_or_cqt,
    )

    dataset = args.dataset or _pick_dataset_interactive()
    feature = args.feature or _pick_feature_interactive()

    # task_mode can be explicit; otherwise infer from checkpoint later.
    provisional_task = args.task_mode or "multi_label"
    weights_run = args.weights_run or _default_weights_run(dataset, feature, provisional_task)
    test_manifest = args.test_manifest or _default_manifest(dataset)

    if args.weights_path:
        weights_path = _resolve_path(repo_root, args.weights_path)
    elif args.weights_run:
        weights_path = _resolve_path(
            repo_root, f"src/models/saved_weights/{weights_run}/best_val.pt"
        )
    else:
        weights_path = _auto_pick_weights_path(repo_root, dataset, feature)
        # Update display name to the picked folder for clearer logs.
        try:
            weights_run = weights_path.parent.parent.name
        except Exception:
            pass

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    # If not specified, infer true task mode from checkpoint metadata.
    task_mode = args.task_mode or _infer_task_mode_from_ckpt(weights_path)
    if task_mode not in {"single_label", "multi_label"}:
        raise ValueError(f"Unsupported task_mode in checkpoint/args: {task_mode}")
    backbone = args.backbone or _infer_backbone_from_ckpt(weights_path)
    model_cls = _resolve_model_cls(backbone)

    test_manifest_path = _resolve_path(repo_root, test_manifest)
    if not test_manifest_path.exists():
        raise FileNotFoundError(f"Test manifest not found: {test_manifest_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Test mode: dataset={dataset}, feature={feature}, task_mode={task_mode}, "
        f"backbone={backbone}, weights_run={weights_run}"
    )

    if feature == "mel":
        preds_arr, gts_arr, sample_ids, _, valid_labels, _ = run_inference_mel(
            model_cls=model_cls,
            model_kwargs={"in_ch": 2},
            model_weights_path=weights_path,
            device=device,
            test_manifest_csv=test_manifest_path,
            root=repo_root,
            show_progress=not args.no_progress,
            task_mode=task_mode,
        )
    elif feature == "cqt":
        preds_arr, gts_arr, sample_ids, _, valid_labels, _ = run_inference_mel_cqt_or_cqt(
            model_cls=model_cls,
            model_kwargs={"in_ch": 2},
            model_weights_path=weights_path,
            device=device,
            test_manifest_csv=test_manifest_path,
            root=repo_root,
            show_progress=not args.no_progress,
            cqt_bins=args.cqt_bins,
            bins_per_octave=args.bins_per_octave,
            feature_mode="cqt",
            task_mode=task_mode,
        )
    else:
        preds_arr, gts_arr, sample_ids, _, valid_labels, _ = run_inference_mel_cqt_or_cqt(
            model_cls=model_cls,
            model_kwargs={"in_ch": 4},
            model_weights_path=weights_path,
            device=device,
            test_manifest_csv=test_manifest_path,
            root=repo_root,
            show_progress=not args.no_progress,
            cqt_bins=args.cqt_bins,
            bins_per_octave=args.bins_per_octave,
            feature_mode="mel_cqt",
            task_mode=task_mode,
        )

    if task_mode == "single_label":
        y_true, y_pred = _evaluate_single_label(preds_arr, gts_arr, valid_labels)
        try:
            out_path = _save_single_label_eval_plot(
                y_true=y_true,
                y_pred=y_pred,
                labels=valid_labels,
                feature=feature,
                ckpt_path=weights_path,
            )
            print(f"Saved test plot: {out_path}")
        except Exception as exc:
            print(f"[WARN] Failed to save single-label test plot: {exc}")
        return

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
    try:
        out_path = _save_multilabel_eval_plot(
            preds_arr=preds_arr,
            gts_arr=gts_arr,
            feature=feature,
            ckpt_path=weights_path,
        )
        print(f"Saved test plot: {out_path}")
    except Exception as exc:
        print(f"[WARN] Failed to save multi-label test plot: {exc}")


if __name__ == "__main__":
    main()
