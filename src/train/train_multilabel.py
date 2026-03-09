"""
Notebook-style training entrypoint for multi-label models.
Edit constants below, then run:

python -m src.train.train_multilabel
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import src
import yaml

from src.data_loader import MultiLabelMelDataset
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
    "best_val_micro_f1",
    "best_val_macro_f1",
    "best_val_exact_match",
    "final_val_micro_f1",
    "final_val_macro_f1",
    "final_val_exact_match",
    "batch_size",
    "lr",
    "weight_decay",
    "dropout",
    "val_frac",
    "seed",
    "threshold",
    "manifests",
    "ckpt_dir",
]


def compute_f1_metrics(
    prob_chunks: List[torch.Tensor],
    target_chunks: List[torch.Tensor],
    threshold: float,
) -> Dict[str, float]:
    """Calculates multi-label metrics using sklearn."""
    from sklearn.metrics import f1_score

    if not prob_chunks:
        return {"micro_f1": 0.0, "macro_f1": 0.0, "exact_match": 0.0}

    y_true = torch.cat(target_chunks).cpu().numpy()
    y_pred = torch.cat(prob_chunks).cpu().numpy()
    y_bin = (y_pred >= threshold).astype(np.int32)

    micro = f1_score(y_true, y_bin, average="micro", zero_division=0)
    macro = f1_score(y_true, y_bin, average="macro", zero_division=0)
    exact = (y_bin == y_true).all(axis=1).mean()

    return {"micro_f1": float(micro), "macro_f1": float(macro), "exact_match": float(exact)}


def train_one_epoch_multi(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: str,
    use_cuda_amp: bool,
    use_mps_amp: bool,
    pin_mem: bool,
    threshold: float,
):
    model.train()
    loss_sum, total = 0.0, 0
    prob_chunks, target_chunks = [], []

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

        prob_chunks.append(torch.sigmoid(logits.detach()).cpu())
        target_chunks.append(y.detach().cpu())
        loss_sum += float(loss.item()) * y.size(0)
        total += int(y.size(0))

    metrics = compute_f1_metrics(prob_chunks, target_chunks, threshold)
    return loss_sum / max(1, total), metrics


def evaluate_multi(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    use_cuda_amp: bool,
    use_mps_amp: bool,
    pin_mem: bool,
    threshold: float,
):
    model.eval()
    loss_sum, total = 0.0, 0
    prob_chunks, target_chunks = [], []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device, non_blocking=pin_mem), y.to(device, non_blocking=pin_mem)
            context = get_autocast_context(use_cuda_amp=use_cuda_amp, use_mps_amp=use_mps_amp)

            with context:
                logits = model(X)
                loss = criterion(logits, y)

            prob_chunks.append(torch.sigmoid(logits.detach()).cpu())
            target_chunks.append(y.detach().cpu())
            loss_sum += float(loss.item()) * y.size(0)
            total += int(y.size(0))

    metrics = compute_f1_metrics(prob_chunks, target_chunks, threshold)
    return loss_sum / max(1, total), metrics


def _normalise_manifests(manifest_csv: Sequence[str] | str) -> List[str]:
    if isinstance(manifest_csv, (list, tuple)):
        return [str(x) for x in manifest_csv]
    return [str(manifest_csv)]


def multi_label_train_loop(
    manifest_csv: Sequence[str] | str,
    classes: Sequence[str],
    ckpt_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    val_frac: float,
    dropout: float,
    patience: int,
    num_workers: int,
    threshold: float,
    seed: int,
    audio_cfg: dict,
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

    dataset = MultiLabelMelDataset(
        manifest_csv=manifests,
        class_names=[c.strip().lower() for c in classes],
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
        num_classes=len(classes),
        dropout=dropout,
        in_ch=in_ch,
        pretrained=pretrained,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    history_keys = [
        "train_loss",
        "val_loss",
        "train_micro_f1",
        "val_micro_f1",
        "train_macro_f1",
        "val_macro_f1",
        "train_exact_match",
        "val_exact_match",
    ]
    history = {k: [] for k in history_keys}
    start_epoch, best_val_f1, no_improve = 1, 0.0, 0

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
        best_val_f1 = ckpt.get("best_val_micro_f1", 0.0)
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
        train_loss, train_m = train_one_epoch_multi(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_cuda_amp=use_cuda_amp,
            use_mps_amp=use_mps_amp,
            pin_mem=pin_mem,
            threshold=threshold,
        )
        val_loss, val_m = evaluate_multi(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_cuda_amp=use_cuda_amp,
            use_mps_amp=use_mps_amp,
            pin_mem=pin_mem,
            threshold=threshold,
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_micro_f1"].append(train_m["micro_f1"])
        history["val_micro_f1"].append(val_m["micro_f1"])
        history["train_macro_f1"].append(train_m["macro_f1"])
        history["val_macro_f1"].append(val_m["macro_f1"])
        history["train_exact_match"].append(train_m["exact_match"])
        history["val_exact_match"].append(val_m["exact_match"])
        write_history_csv(history, Path(ckpt_dir) / "history.csv")

        improved = val_m["micro_f1"] > best_val_f1
        if improved:
            best_val_f1 = val_m["micro_f1"]
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"[{epoch}/{epochs}] "
            f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
            f"Val MicroF1: {val_m['micro_f1']:.4f} | "
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
            "best_val_micro_f1": best_val_f1,
            "no_improve": no_improve,
            "classes": list(classes),
            "audio_config": audio_cfg,
            "label_to_idx": dataset.label_to_idx,
        }
        save_checkpoint(payload, Path(ckpt_dir) / "last.pt")

        if improved:
            save_checkpoint(payload, Path(ckpt_dir) / "best_val.pt")
            if save_best_stamped:
                stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
                save_checkpoint(payload, Path(ckpt_dir) / f"best_val_{stamp}.pt")
        elif no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if history["val_micro_f1"]:
        best_idx = int(np.argmax(history["val_micro_f1"]))
        best_epoch = best_idx + 1
        best_val_macro = float(history["val_macro_f1"][best_idx])
        best_val_exact = float(history["val_exact_match"][best_idx])
    else:
        best_epoch = 0
        best_val_macro = 0.0
        best_val_exact = 0.0

    final_idx = len(history["val_micro_f1"]) - 1
    summary = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "run_name": run_name,
        "model_name": model_name,
        "pretrained": bool(pretrained),
        "freeze_backbone_epochs": int(freeze_backbone_epochs),
        "epochs_trained": len(history["train_loss"]),
        "best_epoch": int(best_epoch),
        "best_val_micro_f1": float(best_val_f1),
        "best_val_macro_f1": best_val_macro,
        "best_val_exact_match": best_val_exact,
        "final_val_micro_f1": float(history["val_micro_f1"][final_idx]) if final_idx >= 0 else 0.0,
        "final_val_macro_f1": float(history["val_macro_f1"][final_idx]) if final_idx >= 0 else 0.0,
        "final_val_exact_match": float(history["val_exact_match"][final_idx]) if final_idx >= 0 else 0.0,
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "dropout": float(dropout),
        "val_frac": float(val_frac),
        "seed": int(seed),
        "threshold": float(threshold),
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
        "classes": list(classes),
        "audio_config": audio_cfg,
        "summary": summary,
    }


RUN_NAME = "multilabel_cnn_v1"
WEIGHTS_DIR = PROJECT_ROOT / "src" / "models" / "saved_weights" / RUN_NAME
MANIFESTS = [
    PROJECT_ROOT / "data" / "processed" / "train_mels.csv",
]
LABELS_YAML = PROJECT_ROOT / "src" / "configs" / "labels.yaml"
AUDIO_CONFIG_YAML = PROJECT_ROOT / "src" / "configs" / "audio_params.yaml"
EXPERIMENT_LOG = PROJECT_ROOT / "src" / "models" / "experiments.csv"

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
    "threshold": 0.5,
    "num_workers": 2,
}


def main() -> None:
    with open(AUDIO_CONFIG_YAML, "r") as f:
        audio_params = yaml.safe_load(f) or {}
    with open(LABELS_YAML, "r") as f:
        label_config = yaml.safe_load(f) or {}

    classes = [c.strip().lower() for c in label_config.get("train_labels", [])]
    if not classes:
        raise ValueError(f"No train_labels found in {LABELS_YAML}")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    resume_ckpt = WEIGHTS_DIR / "last.pt"
    if not resume_ckpt.exists():
        resume_ckpt = None
        print("Starting fresh. No previous weights found.")
    else:
        print(f"Resuming from {resume_ckpt}")

    results = multi_label_train_loop(
        manifest_csv=[str(p) for p in MANIFESTS],
        classes=classes,
        ckpt_dir=WEIGHTS_DIR,
        epochs=TRAIN_CONFIG["epochs"],
        batch_size=TRAIN_CONFIG["batch_size"],
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
        val_frac=TRAIN_CONFIG["val_frac"],
        dropout=TRAIN_CONFIG["dropout"],
        patience=TRAIN_CONFIG["patience"],
        num_workers=TRAIN_CONFIG["num_workers"],
        threshold=TRAIN_CONFIG["threshold"],
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
