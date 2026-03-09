"""
Notebook-style training entrypoint for single-label models.
Edit constants below, then run:

python -m src.train.train_singlelabel
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

import src
import yaml

from src.train.utils import (
    append_experiment_log,
    build_model,
    collate_fn_padd,
    get_autocast_context,
    get_device,
    load_checkpoint,
    load_model_weights,
    save_checkpoint,
    seed_everything,
    set_feature_extractor_trainable,
    write_history_csv,
)

PROJECT_ROOT = Path(src.__file__).resolve().parents[1]

EXPERIMENT_LOG_FIELDS = [
    "timestamp_utc",
    "run_name",
    "model_name",
    "pretrained",
    "freeze_backbone_epochs",
    "epochs_trained",
    "best_epoch",
    "best_val_acc",
    "best_val_macro_f1",
    "final_val_acc",
    "final_val_macro_f1",
    "batch_size",
    "lr",
    "weight_decay",
    "dropout",
    "val_frac",
    "seed",
    "class_source",
    "manifests",
    "ckpt_dir",
]


def _load_manifests(manifest_csv: Sequence[str] | str) -> pd.DataFrame:
    if isinstance(manifest_csv, (list, tuple)):
        if not manifest_csv:
            raise ValueError("manifest_csv list is empty")
        frames = [pd.read_csv(path) for path in manifest_csv]
        return pd.concat(frames, ignore_index=True)
    return pd.read_csv(manifest_csv)


class SingleLabelMelDataset(Dataset):
    def __init__(
        self,
        manifest_csv: Sequence[str] | str,
        class_names: Optional[Sequence[str]] = None,
        project_root: str | Path = ".",
        transform=None,
    ):
        df = _load_manifests(manifest_csv)
        if "filepath" not in df.columns:
            raise ValueError("Manifest must contain a 'filepath' column")
        if "label" not in df.columns and "labels" not in df.columns:
            raise ValueError("Manifest must contain either a 'label' or 'labels' column")

        if "label" in df.columns:
            labels = df["label"]
        else:
            labels = df["labels"]

        labels = labels.fillna("").astype(str).str.lower().str.replace(" ", "", regex=False)
        labels = labels.str.replace(",", "|", regex=False).str.split("|").str[0]

        work = df[["filepath"]].copy()
        work["label"] = labels
        work = work[work["label"] != ""].copy()
        if work.empty:
            raise ValueError("No valid labels found in manifest.")

        inferred_classes = sorted(work["label"].unique().tolist())
        if class_names is None:
            resolved_classes = inferred_classes
        else:
            resolved_classes = [c.strip().lower() for c in class_names if str(c).strip()]
            if not resolved_classes:
                raise ValueError("class_names was provided but empty after normalization.")
            unknown = sorted(set(work["label"].unique()) - set(resolved_classes))
            if unknown:
                preview = ", ".join(unknown[:8])
                raise ValueError(
                    "Manifest contains labels not present in class_names. "
                    f"Unknown examples: {preview}"
                )

        self.class_names = resolved_classes
        self.label_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        if self.num_classes < 2:
            raise ValueError("Single-label training needs at least 2 classes.")

        work["target"] = work["label"].map(self.label_to_idx)
        work = work.dropna(subset=["target"]).copy()
        work["target"] = work["target"].astype(int)
        if work.empty:
            raise ValueError("No rows left after mapping labels to target indices.")

        self.df = work.reset_index(drop=True)
        self.root = Path(project_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        path_str = str(row["filepath"])
        npy_path = Path(path_str) if Path(path_str).is_absolute() else self.root / path_str

        mel = np.load(npy_path)
        mel_tensor = torch.from_numpy(mel).float()

        mean = mel_tensor.mean()
        std = mel_tensor.std() + 1e-6
        mel_tensor = (mel_tensor - mean) / std

        if self.transform is not None:
            mel_tensor = self.transform(mel_tensor)

        target = torch.tensor(int(row["target"]), dtype=torch.long)
        return mel_tensor, target


def compute_single_label_metrics(pred_chunks: List[torch.Tensor], target_chunks: List[torch.Tensor]) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    if not pred_chunks:
        return {"acc": 0.0, "macro_f1": 0.0}

    y_pred = torch.cat(pred_chunks).cpu().numpy()
    y_true = torch.cat(target_chunks).cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"acc": float(acc), "macro_f1": float(macro_f1)}


def train_one_epoch_single(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: str,
    use_cuda_amp: bool,
    use_mps_amp: bool,
    pin_mem: bool,
):
    model.train()
    loss_sum, total = 0.0, 0
    pred_chunks, target_chunks = [], []

    for X, y in loader:
        X, y = X.to(device, non_blocking=pin_mem), y.to(device, non_blocking=pin_mem)
        optimizer.zero_grad(set_to_none=True)

        context = get_autocast_context(use_cuda_amp=use_cuda_amp, use_mps_amp=use_mps_amp)
        with context:
            logits = model(X)
            loss = criterion(logits, y)

        if use_cuda_amp:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        pred_chunks.append(torch.argmax(logits.detach(), dim=1).cpu())
        target_chunks.append(y.detach().cpu())
        loss_sum += float(loss.item()) * y.size(0)
        total += int(y.size(0))

    metrics = compute_single_label_metrics(pred_chunks, target_chunks)
    return loss_sum / max(1, total), metrics


def evaluate_single(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    use_cuda_amp: bool,
    use_mps_amp: bool,
    pin_mem: bool,
):
    model.eval()
    loss_sum, total = 0.0, 0
    pred_chunks, target_chunks = [], []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device, non_blocking=pin_mem), y.to(device, non_blocking=pin_mem)
            context = get_autocast_context(use_cuda_amp=use_cuda_amp, use_mps_amp=use_mps_amp)

            with context:
                logits = model(X)
                loss = criterion(logits, y)

            pred_chunks.append(torch.argmax(logits.detach(), dim=1).cpu())
            target_chunks.append(y.detach().cpu())
            loss_sum += float(loss.item()) * y.size(0)
            total += int(y.size(0))

    metrics = compute_single_label_metrics(pred_chunks, target_chunks)
    return loss_sum / max(1, total), metrics


def _normalise_manifests(manifest_csv: Sequence[str] | str) -> List[str]:
    if isinstance(manifest_csv, (list, tuple)):
        return [str(x) for x in manifest_csv]
    return [str(manifest_csv)]


def single_label_train_loop(
    manifest_csv: Sequence[str] | str,
    ckpt_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    val_frac: float,
    dropout: float,
    patience: int,
    num_workers: int,
    seed: int,
    audio_cfg: dict,
    classes: Optional[Sequence[str]] = None,
    class_source: str = "manifest",
    resume_from: Optional[Path] = None,
    save_best_stamped: bool = False,
    model_name: str = "cnn",
    in_ch: int = 2,
    pretrained: bool = False,
    freeze_backbone_epochs: int = 0,
    warm_start_from: Optional[Path] = None,
    run_name: Optional[str] = None,
    experiment_log: Optional[Path] = None,
):
    seed_everything(seed)
    device, use_cuda_amp, use_mps_amp, scaler, pin_mem = get_device()
    run_name = run_name or Path(ckpt_dir).name
    manifests = _normalise_manifests(manifest_csv)

    dataset = SingleLabelMelDataset(
        manifest_csv=manifests,
        class_names=classes,
        project_root=PROJECT_ROOT,
    )
    if len(dataset) < 2:
        raise ValueError("Dataset must contain at least 2 examples for train/validation split.")

    val_size = int(round(len(dataset) * val_frac))
    val_size = min(max(1, val_size), len(dataset) - 1)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        collate_fn=collate_fn_padd,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        collate_fn=collate_fn_padd,
    )

    model = build_model(
        model_name=model_name,
        num_classes=dataset.num_classes,
        dropout=dropout,
        in_ch=in_ch,
        pretrained=pretrained,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history_keys = [
        "train_loss",
        "val_loss",
        "train_acc",
        "val_acc",
        "train_macro_f1",
        "val_macro_f1",
    ]
    history = {k: [] for k in history_keys}
    start_epoch, best_val_acc, no_improve = 1, 0.0, 0

    if warm_start_from and warm_start_from.exists():
        load_model_weights(warm_start_from, device, model)
        print(f"Warm-started model weights from: {warm_start_from}")

    if resume_from and resume_from.exists():
        ckpt = load_checkpoint(
            resume_from,
            device,
            model,
            optimizer,
            scheduler,
            scaler,
            current_label_to_idx=dataset.label_to_idx,
        )
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", history)
        for key in history_keys:
            history.setdefault(key, [])
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        no_improve = ckpt.get("no_improve", 0)

    feature_is_frozen = False
    if freeze_backbone_epochs > 0 and start_epoch <= freeze_backbone_epochs:
        feature_is_frozen = set_feature_extractor_trainable(model, trainable=False)
        if feature_is_frozen:
            print(f"Feature extractor frozen for epochs 1..{freeze_backbone_epochs}")
        else:
            print("freeze_backbone_epochs ignored: selected model does not expose freeze hooks.")

    for epoch in range(start_epoch, epochs + 1):
        if feature_is_frozen and epoch == freeze_backbone_epochs + 1:
            set_feature_extractor_trainable(model, trainable=True)
            feature_is_frozen = False
            print(f"Feature extractor unfrozen at epoch {epoch}")

        t0 = time.time()
        train_loss, train_m = train_one_epoch_single(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_cuda_amp=use_cuda_amp,
            use_mps_amp=use_mps_amp,
            pin_mem=pin_mem,
        )
        val_loss, val_m = evaluate_single(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_cuda_amp=use_cuda_amp,
            use_mps_amp=use_mps_amp,
            pin_mem=pin_mem,
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_m["acc"])
        history["val_acc"].append(val_m["acc"])
        history["train_macro_f1"].append(train_m["macro_f1"])
        history["val_macro_f1"].append(val_m["macro_f1"])
        write_history_csv(history, Path(ckpt_dir) / "history.csv")

        improved = val_m["acc"] > best_val_acc
        if improved:
            best_val_acc = val_m["acc"]
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"[{epoch}/{epochs}] "
            f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
            f"Val Acc: {val_m['acc']:.4f} | "
            f"Val MacroF1: {val_m['macro_f1']:.4f} | "
            f"Time: {time.time()-t0:.1f}s"
        )

        payload = {
            "epoch": epoch,
            "model_name": model_name,
            "pretrained": pretrained,
            "freeze_backbone_epochs": freeze_backbone_epochs,
            "model_state": model.state_dict(),
            "opt_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if scaler and scaler.is_enabled() else None,
            "history": history,
            "best_val_acc": best_val_acc,
            "no_improve": no_improve,
            "classes": dataset.class_names,
            "audio_config": audio_cfg,
            "label_to_idx": dataset.label_to_idx,
        }
        save_checkpoint(payload, Path(ckpt_dir) / "last.pt")

        if improved:
            save_checkpoint(payload, Path(ckpt_dir) / "best_val_acc.pt")
            if save_best_stamped:
                stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
                save_checkpoint(payload, Path(ckpt_dir) / f"best_val_acc_{stamp}.pt")
        elif no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if history["val_acc"]:
        best_idx = int(np.argmax(history["val_acc"]))
        best_epoch = best_idx + 1
        best_val_macro = float(history["val_macro_f1"][best_idx])
    else:
        best_epoch = 0
        best_val_macro = 0.0

    final_idx = len(history["val_acc"]) - 1
    summary = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "run_name": run_name,
        "model_name": model_name,
        "pretrained": bool(pretrained),
        "freeze_backbone_epochs": int(freeze_backbone_epochs),
        "epochs_trained": len(history["train_loss"]),
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "best_val_macro_f1": best_val_macro,
        "final_val_acc": float(history["val_acc"][final_idx]) if final_idx >= 0 else 0.0,
        "final_val_macro_f1": float(history["val_macro_f1"][final_idx]) if final_idx >= 0 else 0.0,
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "dropout": float(dropout),
        "val_frac": float(val_frac),
        "seed": int(seed),
        "class_source": class_source,
        "manifests": "|".join(manifests),
        "ckpt_dir": str(Path(ckpt_dir).resolve()),
    }
    with open(Path(ckpt_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if experiment_log is not None:
        append_experiment_log(Path(experiment_log), summary, EXPERIMENT_LOG_FIELDS)

    return {
        "model": model,
        "history": history,
        "classes": dataset.class_names,
        "audio_config": audio_cfg,
        "summary": summary,
    }


RUN_NAME = "CNN_singlelabel_v1"
WEIGHTS_DIR = PROJECT_ROOT / "src" / "models" / "saved_weights" / RUN_NAME
MANIFESTS = [
    PROJECT_ROOT / "data" / "processed" / "train_mels.csv",
]
LABELS_YAML = PROJECT_ROOT / "src" / "configs" / "labels.yaml"
AUDIO_CONFIG_YAML = PROJECT_ROOT / "src" / "configs" / "audio_params.yaml"
EXPERIMENT_LOG = PROJECT_ROOT / "src" / "models" / "experiments_singlelabel.csv"
LABEL_KEY: Optional[str] = None  # e.g. "irmas" to force a known class ordering

MODEL_CONFIG = {
    "model_name": "cnn",
    "in_ch": 2,
    "pretrained": False,
    "freeze_backbone_epochs": 0,
}

TRAIN_CONFIG = {
    "batch_size": 32,
    "lr": 1e-3,
    "epochs": 120,
    "patience": 20,
    "weight_decay": 1e-4,
    "dropout": 0.5,
    "val_frac": 0.2,
    "seed": 1337,
    "num_workers": 0,
}


def main() -> None:
    with open(AUDIO_CONFIG_YAML, "r") as f:
        audio_params = yaml.safe_load(f) or {}
    with open(LABELS_YAML, "r") as f:
        label_config = yaml.safe_load(f) or {}

    classes = None
    class_source = "manifest"
    if LABEL_KEY is not None:
        classes = [c.strip().lower() for c in label_config.get(LABEL_KEY, [])]
        if not classes:
            raise ValueError(f"No labels found for key '{LABEL_KEY}' in {LABELS_YAML}")
        class_source = f"labels.yaml:{LABEL_KEY}"

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    resume_ckpt = WEIGHTS_DIR / "last.pt"
    if not resume_ckpt.exists():
        resume_ckpt = None
        print("Starting fresh. No previous weights found.")
    else:
        print(f"Resuming from {resume_ckpt}")

    results = single_label_train_loop(
        manifest_csv=[str(p) for p in MANIFESTS],
        classes=classes,
        class_source=class_source,
        ckpt_dir=WEIGHTS_DIR,
        epochs=TRAIN_CONFIG["epochs"],
        batch_size=TRAIN_CONFIG["batch_size"],
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
        val_frac=TRAIN_CONFIG["val_frac"],
        dropout=TRAIN_CONFIG["dropout"],
        patience=TRAIN_CONFIG["patience"],
        num_workers=TRAIN_CONFIG["num_workers"],
        seed=TRAIN_CONFIG["seed"],
        audio_cfg=audio_params.get("audio", audio_params),
        resume_from=resume_ckpt,
        model_name=MODEL_CONFIG["model_name"],
        in_ch=MODEL_CONFIG["in_ch"],
        pretrained=MODEL_CONFIG["pretrained"],
        freeze_backbone_epochs=MODEL_CONFIG["freeze_backbone_epochs"],
        run_name=RUN_NAME,
        experiment_log=EXPERIMENT_LOG,
    )

    print("Training complete.")
    print("Summary:", results["summary"])


if __name__ == "__main__":
    main()
