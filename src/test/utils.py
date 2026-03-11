from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import matplotlib
try:
    from IPython import get_ipython
except Exception:  # pragma: no cover - optional dependency/runtime
    get_ipython = None

if os.environ.get("MPLBACKEND") is None:
    ip = get_ipython() if get_ipython is not None else None
    if ip is None or "IPKernelApp" not in getattr(ip, "config", {}):
        matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)

from src.train.metrics import compute_multi_label_metrics, compute_single_label_metrics
from src.preprocessing.features import compute_stft_params, compute_stereo_logmel_db
from src.preprocessing.preprocessing import (
    conform_audio_duration,
    ensure_directory_exists,
    load_audio_as_stereo_and_resample,
    preprocess_loudness,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IRMAS_MEL_AUDIO_KEYS = (
    "sr",
    "duration",
    "n_mels",
    "win_ms",
    "hop_ms",
    "fmin",
    "fmax",
    "window",
    "loudness_norm",
    "target_lufs",
    "loudness_peak_limit",
)


def _to_builtin(value):
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _resolve_audio_section(audio_cfg: dict) -> dict:
    return (audio_cfg.get("audio", audio_cfg) or {}) if isinstance(audio_cfg, dict) else {}


def _path_to_repo_string(path: Path, project_root: Path) -> str:
    path = Path(path)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    else:
        path = path.resolve()
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def irmas_mel_cache_dir(cache_root: Path, audio_cfg: dict, class_names: Sequence[str]) -> Path:
    audio = _resolve_audio_section(audio_cfg)
    payload = {
        "audio": {key: audio.get(key) for key in IRMAS_MEL_AUDIO_KEYS},
        "classes": [str(name).strip().lower() for name in class_names],
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return Path(cache_root) / f"irmas_part1_mel_{digest}"


def parse_irmas_sidecar_labels(txt_path: Path, allowed_labels: Sequence[str]) -> list[str]:
    allowed = {str(label).strip().lower() for label in allowed_labels}
    labels: list[str] = []
    for raw_line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        label = raw_line.strip().lower()
        if label and label in allowed and label not in labels:
            labels.append(label)
    return labels


def extract_cached_mel_feature(wav_path: Path, out_path: Path, audio_cfg: dict) -> Path:
    audio = _resolve_audio_section(audio_cfg)
    sr = int(audio["sr"])
    duration = float(audio["duration"])

    stereo = load_audio_as_stereo_and_resample(wav_path, target_sr=sr)
    stereo = conform_audio_duration(stereo, sr, duration)
    stereo = preprocess_loudness(
        stereo,
        sr=sr,
        loudness_norm=audio.get("loudness_norm", "none"),
        target_lufs=audio.get("target_lufs", -23.0),
        peak_limit=audio.get("loudness_peak_limit", 0.99),
    )
    n_fft, hop, win_length = compute_stft_params(sr, audio["win_ms"], audio["hop_ms"])
    mel = compute_stereo_logmel_db(
        stereo,
        sr,
        n_fft=n_fft,
        hop=hop,
        win_length=win_length,
        n_mels=audio["n_mels"],
        fmin=audio["fmin"],
        fmax=audio.get("fmax"),
        window=str(audio.get("window", "hann")),
    )
    ensure_directory_exists(out_path.parent)
    np.save(out_path, mel.astype(np.float32))
    return out_path


def prepare_irmas_part1_mel_manifest(
    test_root: Path,
    cache_root: Path,
    audio_cfg: dict,
    class_names: Sequence[str],
    *,
    force_rebuild: bool = False,
    project_root: Path | None = None,
) -> Path:
    project_root = PROJECT_ROOT if project_root is None else Path(project_root).resolve()
    test_root = Path(test_root).resolve()
    if not test_root.exists():
        raise FileNotFoundError(f"IRMAS test root not found: {test_root}")

    cache_dir = irmas_mel_cache_dir(Path(cache_root), audio_cfg, class_names)
    features_dir = cache_dir / "features"
    manifest_path = cache_dir / "irmas_test_part1_mels.csv"
    if manifest_path.exists() and not force_rebuild:
        return manifest_path

    rows: list[list[str]] = []
    missing_txt = 0
    missing_labels = 0

    for wav_path in sorted(test_root.glob("*.wav")):
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            missing_txt += 1
            continue

        labels = parse_irmas_sidecar_labels(txt_path, class_names)
        if not labels:
            missing_labels += 1
            continue

        digest = hashlib.sha1(str(wav_path).encode("utf-8")).hexdigest()[:10]
        feature_path = features_dir / f"{wav_path.stem}__{digest}.npy"
        if force_rebuild or not feature_path.exists():
            extract_cached_mel_feature(wav_path, feature_path, audio_cfg)

        rows.append(
            [
                _path_to_repo_string(feature_path, project_root),
                labels[0],
                _path_to_repo_string(wav_path, project_root),
                _path_to_repo_string(txt_path, project_root),
                "|".join(labels),
            ]
        )

    if not rows:
        raise ValueError(
            f"No IRMAS test samples with valid sidecar labels were found under {test_root}"
        )

    ensure_directory_exists(manifest_path.parent)
    pd.DataFrame(rows, columns=["filepath", "label", "wavpath", "txt_path", "all_labels"]).to_csv(
        manifest_path, index=False
    )

    if missing_txt or missing_labels:
        print(
            f"[INFO] IRMAS Part1 manifest built with {len(rows)} samples. "
            f"Skipped {missing_txt} without .txt and {missing_labels} without valid labels."
        )
    else:
        print(f"[INFO] IRMAS Part1 manifest built with {len(rows)} samples.")

    return manifest_path


def build_single_label_prediction_table(
    y_pred: np.ndarray,
    dataset_df: pd.DataFrame,
    class_names: Sequence[str],
    *,
    project_root: Path | None = None,
) -> pd.DataFrame:
    project_root = PROJECT_ROOT if project_root is None else Path(project_root).resolve()
    if len(y_pred) != len(dataset_df):
        raise ValueError(
            f"Prediction count ({len(y_pred)}) does not match dataset rows ({len(dataset_df)})"
        )

    predicted_labels = [class_names[int(idx)] for idx in y_pred]
    true_labels = dataset_df["label"].astype(str).str.lower().tolist()
    all_labels_series = dataset_df.get("all_labels", dataset_df["label"]).fillna("").astype(str)
    all_labels_list = [
        [label for label in raw.lower().split("|") if label]
        for raw in all_labels_series.tolist()
    ]

    return pd.DataFrame(
        {
            "wav_path": [
                _path_to_repo_string(Path(path), project_root)
                for path in dataset_df["wavpath"].tolist()
            ],
            "feature_path": [
                _path_to_repo_string(Path(path), project_root)
                for path in dataset_df["filepath"].tolist()
            ],
            "primary_true_label": true_labels,
            "all_true_labels": ["|".join(labels) for labels in all_labels_list],
            "predicted_label": predicted_labels,
            "primary_match": [pred == true for pred, true in zip(predicted_labels, true_labels)],
            "hit_any_label": [
                pred in labels if labels else False
                for pred, labels in zip(predicted_labels, all_labels_list)
            ],
        }
    )


def prepare_targets(y_raw: torch.Tensor, task_mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Match training-time target handling for evaluation."""
    mode = str(task_mode).strip().lower()
    if mode == "multi_label":
        valid_mask = torch.ones(y_raw.size(0), dtype=torch.bool, device=y_raw.device)
        return y_raw.float(), valid_mask

    if y_raw.ndim == 1:
        valid_mask = torch.ones(y_raw.size(0), dtype=torch.bool, device=y_raw.device)
        return y_raw.long(), valid_mask

    sums = y_raw.sum(dim=1)
    valid_mask = sums > 0
    y_idx = torch.argmax(y_raw, dim=1).long()
    return y_idx, valid_mask


def choose_subset_indices(
    dataset_size: int,
    *,
    split: str,
    split_indices_path: Path | None,
    val_frac: float,
    seed: int,
) -> list[int]:
    split = str(split).strip().lower()
    if split == "full":
        return list(range(dataset_size))

    if split_indices_path is not None and split_indices_path.exists():
        saved = torch.load(split_indices_path, map_location="cpu", weights_only=False)
        key = f"{split}_indices"
        if key not in saved:
            raise KeyError(f"Split file does not contain '{key}': {split_indices_path}")
        return [int(i) for i in saved[key]]

    if dataset_size < 2:
        return list(range(dataset_size))

    val_frac = min(max(float(val_frac), 0.01), 0.9)
    n_val = max(1, int(round(dataset_size * val_frac)))
    n_train = dataset_size - n_val
    if n_train <= 0:
        n_train, n_val = dataset_size - 1, 1

    gen = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(dataset_size, generator=gen).tolist()
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    return train_idx if split == "train" else val_idx


def run_model_predictions(
    model: torch.nn.Module,
    loader,
    *,
    device: str,
    task_mode: str,
    threshold: float,
    pin_mem: bool,
) -> dict:
    model.eval()
    y_true_chunks: list[np.ndarray] = []
    y_pred_chunks: list[np.ndarray] = []
    y_prob_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for x, y_raw in loader:
            x = x.to(device, non_blocking=pin_mem)
            y_raw = y_raw.to(device, non_blocking=pin_mem)

            y_target, valid_mask = prepare_targets(y_raw, task_mode)
            if not valid_mask.any():
                continue

            x_valid = x[valid_mask]
            y_valid = y_target[valid_mask]
            logits = model(x_valid)

            if task_mode == "single_label":
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                y_true_chunks.append(y_valid.detach().cpu().numpy().astype(np.int64))
                y_pred_chunks.append(preds.detach().cpu().numpy().astype(np.int64))
                y_prob_chunks.append(probs.detach().cpu().numpy().astype(np.float32))
            else:
                probs = torch.sigmoid(logits)
                preds = (probs >= float(threshold)).to(torch.int32)
                y_true_chunks.append(y_valid.detach().cpu().numpy().astype(np.int32))
                y_pred_chunks.append(preds.detach().cpu().numpy().astype(np.int32))
                y_prob_chunks.append(probs.detach().cpu().numpy().astype(np.float32))

    if not y_true_chunks:
        raise ValueError("Evaluation produced no valid samples.")

    return {
        "y_true": np.concatenate(y_true_chunks, axis=0),
        "y_pred": np.concatenate(y_pred_chunks, axis=0),
        "y_prob": np.concatenate(y_prob_chunks, axis=0),
    }


def build_summary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, *, task_mode: str, threshold: float) -> dict:
    if task_mode == "single_label":
        return compute_single_label_metrics(y_true.astype(np.int64), y_pred.astype(np.int64))
    return compute_multi_label_metrics(y_true.astype(np.int32), y_prob.astype(np.float32), threshold=float(threshold))


def per_class_single_label_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str]) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "class_name": list(class_names),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def per_class_multi_label_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    *,
    threshold: float,
) -> pd.DataFrame:
    y_pred = (y_prob >= float(threshold)).astype(np.int32)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "class_name": list(class_names),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def build_detailed_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    *,
    task_mode: str,
    threshold: float,
) -> dict:
    summary = build_summary_metrics(y_true, y_pred, y_prob, task_mode=task_mode, threshold=threshold)

    if task_mode == "single_label":
        per_class_df = per_class_single_label_metrics(y_true, y_pred, class_names)
        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=list(class_names),
            output_dict=True,
            zero_division=0,
        )
    else:
        per_class_df = per_class_multi_label_metrics(y_true, y_prob, class_names, threshold=threshold)
        report = classification_report(
            y_true,
            (y_prob >= float(threshold)).astype(np.int32),
            target_names=list(class_names),
            output_dict=True,
            zero_division=0,
        )

    return {
        "summary": summary,
        "per_class": per_class_df,
        "report": report,
    }


def resolve_checkpoint_path(model_path: str | Path, *, project_root: Path | None = None) -> Path:
    project_root = PROJECT_ROOT if project_root is None else Path(project_root).resolve()
    path = Path(model_path)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    else:
        path = path.resolve()

    checkpoint_path = path / "best_val.pt" if path.is_dir() else path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def history_dict_to_frame(history: dict) -> pd.DataFrame:
    if not isinstance(history, dict) or not history:
        raise ValueError("History payload is empty or invalid.")

    history_lists = {
        str(key): [float(v) if v is not None else np.nan for v in values]
        for key, values in history.items()
        if isinstance(values, list)
    }
    if not history_lists:
        raise ValueError("History payload does not contain any list-valued metrics.")

    n_rows = max(len(values) for values in history_lists.values())
    frame_data = {"epoch": list(range(1, n_rows + 1))}
    for key, values in history_lists.items():
        padded = list(values) + [np.nan] * (n_rows - len(values))
        frame_data[key] = padded
    return pd.DataFrame(frame_data)


def load_training_history_frame(
    model_path: str | Path,
    *,
    project_root: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    checkpoint_path = resolve_checkpoint_path(model_path, project_root=project_root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    history = checkpoint.get("history")
    if not isinstance(history, dict) or not history:
        csv_path = checkpoint_path.parent / "history.csv"
        if not csv_path.exists():
            raise ValueError(f"No training history found in checkpoint or CSV for: {checkpoint_path}")
        history_df = pd.read_csv(csv_path)
    else:
        history_df = history_dict_to_frame(history)

    best_epoch = None
    if "val_loss" in history_df.columns and history_df["val_loss"].notna().any():
        best_epoch = int(history_df.loc[history_df["val_loss"].idxmin(), "epoch"])

    metadata: dict[str, object] = {
        "checkpoint_path": str(checkpoint_path),
        "run_dir": str(checkpoint_path.parent),
        "saved_epoch": int(checkpoint.get("epoch", len(history_df))),
        "best_val_loss": float(checkpoint["best_val_loss"]) if checkpoint.get("best_val_loss") is not None else np.nan,
        "epochs_no_improve": int(checkpoint.get("epochs_no_improve", 0)),
        "best_epoch": best_epoch,
    }
    return history_df, metadata


def build_training_loss_figure(history_df: pd.DataFrame) -> Figure:
    if "epoch" not in history_df.columns:
        raise ValueError("history_df must include an 'epoch' column.")

    columns = [col for col in ("train_loss", "val_loss") if col in history_df.columns]
    if not columns:
        raise ValueError("history_df does not contain train_loss or val_loss.")

    fig, ax = plt.subplots(figsize=(10, 5))
    label_map = {"train_loss": "Train Loss", "val_loss": "Validation Loss"}
    for col in columns:
        ax.plot(history_df["epoch"], history_df[col], marker="o", linewidth=2, label=label_map[col])

    if "val_loss" in history_df.columns and history_df["val_loss"].notna().any():
        best_idx = history_df["val_loss"].idxmin()
        best_epoch = history_df.loc[best_idx, "epoch"]
        best_val = history_df.loc[best_idx, "val_loss"]
        ax.scatter([best_epoch], [best_val], color="crimson", zorder=3, label="Best Val Loss")

    ax.set_title("Training Loss Convergence")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def build_training_accuracy_figure(history_df: pd.DataFrame) -> Figure:
    if "epoch" not in history_df.columns:
        raise ValueError("history_df must include an 'epoch' column.")

    columns = [col for col in ("train_acc", "val_acc") if col in history_df.columns]
    if not columns:
        raise ValueError("history_df does not contain train_acc or val_acc.")

    fig, ax = plt.subplots(figsize=(10, 5))
    label_map = {"train_acc": "Train Accuracy", "val_acc": "Validation Accuracy"}
    for col in columns:
        ax.plot(history_df["epoch"], history_df[col], marker="o", linewidth=2, label=label_map[col])

    if history_df[columns].max().max() <= 1.05:
        ax.set_ylim(0.0, 1.0)

    ax.set_title("Training Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def save_single_label_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    out_path: Path,
) -> None:
    fig = build_single_label_confusion_matrix_figure(y_true, y_pred, class_names)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_single_label_confusion_matrix_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> Figure:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ticks = list(range(len(class_names)))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def save_multi_label_confusion_matrices(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    out_path: Path,
    *,
    threshold: float,
) -> None:
    fig = build_multi_label_confusion_matrices_figure(
        y_true,
        y_prob,
        class_names,
        threshold=threshold,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_multi_label_confusion_matrices_figure(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    *,
    threshold: float,
) -> Figure:
    y_pred = (y_prob >= float(threshold)).astype(np.int32)
    cms = multilabel_confusion_matrix(y_true, y_pred)

    n_classes = len(class_names)
    n_cols = min(3, max(1, n_classes))
    n_rows = int(np.ceil(n_classes / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.0 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for idx, (cm, class_name) in enumerate(zip(cms, class_names)):
        ax = axes[idx]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(class_name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1], ["0", "1"])
        ax.set_yticks([0, 1], ["0", "1"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=9)

    for ax in axes[n_classes:]:
        ax.axis("off")

    fig.colorbar(im, ax=axes[:n_classes].tolist(), fraction=0.02, pad=0.02)
    fig.suptitle("Per-Class Confusion Matrices", y=1.02)
    fig.tight_layout()
    return fig


def save_single_label_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    out_path: Path,
) -> None:
    fig = build_single_label_roc_curves_figure(y_true, y_prob, class_names)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_single_label_roc_curves_figure(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
) -> Figure:
    y_true_ovr = np.eye(len(class_names), dtype=np.int32)[y_true.astype(np.int64)]
    fig, ax = plt.subplots(figsize=(10, 8))
    plotted = 0

    for idx, class_name in enumerate(class_names):
        y_true_cls = y_true_ovr[:, idx]
        if np.unique(y_true_cls).size < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true_cls, y_prob[:, idx])
        ax.plot(fpr, tpr, label=f"{class_name} (AUC={auc(fpr, tpr):.3f})")
        plotted += 1

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title("One-vs-Rest ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    if plotted:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def save_multi_label_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    out_path: Path,
) -> None:
    fig = build_multi_label_roc_curves_figure(y_true, y_prob, class_names)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_multi_label_roc_curves_figure(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 8))
    plotted = 0

    for idx, class_name in enumerate(class_names):
        y_true_cls = y_true[:, idx]
        if np.unique(y_true_cls).size < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true_cls, y_prob[:, idx])
        ax.plot(fpr, tpr, label=f"{class_name} (AUC={auc(fpr, tpr):.3f})")
        plotted += 1

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title("Per-Class ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    if plotted:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def build_evaluation_plot_figures(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    *,
    task_mode: str,
    threshold: float,
) -> dict[str, Figure]:
    if task_mode == "single_label":
        return {
            "confusion_matrix": build_single_label_confusion_matrix_figure(y_true, y_pred, class_names),
            "roc_curves": build_single_label_roc_curves_figure(y_true, y_prob, class_names),
        }

    return {
        "confusion_matrices": build_multi_label_confusion_matrices_figure(
            y_true,
            y_prob,
            class_names,
            threshold=threshold,
        ),
        "roc_curves": build_multi_label_roc_curves_figure(y_true, y_prob, class_names),
    }

def save_report_files(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_class_df = report["per_class"]
    per_class_df.to_csv(out_dir / "per_class_metrics.csv", index=False)

    with open(out_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(_to_builtin(report["summary"]), f, indent=2, sort_keys=True)

    with open(out_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(_to_builtin(report["report"]), f, indent=2, sort_keys=True)
