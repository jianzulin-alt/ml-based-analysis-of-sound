import os
import time
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, random_split


def seed_everything(seed: int = 1337):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    use_cuda_amp = False
    use_mps_amp = False
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    pin_mem = False

    if torch.cuda.is_available():
        device = "cuda"
        use_cuda_amp = True
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        pin_mem = True
    elif torch.backends.mps.is_available():
        device = "mps"
        use_mps_amp = True
    else:
        device = "cpu"

    return device, use_cuda_amp, use_mps_amp, scaler, pin_mem


def _resolve_model_class(backbone: str):
    name = str(backbone).strip().lower()
    if name == "crnn":
        from src.models.CRNN import CRNN

        return CRNN
    if name in {"densenet121", "densenet_121", "cnn_densenet_121"}:
        from src.models.CNN_DenseNet_121 import CNN_DenseNet_121

        return CNN_DenseNet_121
    raise ValueError(f"Unsupported backbone: {backbone}")


def build_model(
    num_classes: int,
    dropout: float,
    in_ch: int,
    freq_bins: int,
    device: str = "cpu",
    backbone: str = "densenet121",
):
    model_cls = _resolve_model_class(backbone)
    model = model_cls(in_ch=in_ch, num_classes=num_classes, p_drop=dropout, freq_bins=freq_bins)
    return model.to(device)


def infer_input_shape(dataset) -> tuple[int, int]:
    if len(dataset) == 0:
        raise ValueError("Dataset is empty; cannot infer input shape.")
    sample_x, _ = dataset[0]
    if not isinstance(sample_x, torch.Tensor):
        sample_x = torch.as_tensor(sample_x)
    if sample_x.dim() != 3:
        raise ValueError(f"Expected sample shape (C, H, W), got {tuple(sample_x.shape)}")
    in_ch = int(sample_x.shape[0])
    freq_bins = int(sample_x.shape[1])
    return in_ch, freq_bins


def save_checkpoint(payload: Dict[str, Any], filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, filepath)


def load_checkpoint(
    path: Path,
    device: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["opt_state"])

    if scheduler and "sched_state" in ckpt:
        scheduler.load_state_dict(ckpt["sched_state"])

    if scaler and ckpt.get("scaler_state") and scaler.is_enabled():
        scaler.load_state_dict(ckpt["scaler_state"])

    return ckpt


def collate_fn_padd(batch):
    tensors = [item[0] for item in batch]
    targets = torch.stack([item[1] for item in batch])

    tensors = [t.permute(2, 0, 1) for t in tensors]
    tensors_padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    tensors_padded = tensors_padded.permute(0, 2, 3, 1)
    return tensors_padded, targets


def _single_targets_from_multi_hot(y_multi: torch.Tensor):
    """
    Convert multi-hot labels into class indices for CE training.
    - zero-label rows are ignored
    - multi-positive rows use argmax (first highest index by value)
    """
    sums = y_multi.sum(dim=1)
    valid_mask = sums > 0
    multi_pos_mask = sums > 1
    y_idx = torch.argmax(y_multi, dim=1).long()
    return y_idx, valid_mask, multi_pos_mask


def _compute_single_metrics(pred_chunks: List[torch.Tensor], target_chunks: List[torch.Tensor]) -> Dict[str, float]:
    if not pred_chunks:
        return {"accuracy": 0.0, "macro_f1": 0.0}

    y_pred = torch.cat(pred_chunks).cpu().numpy()
    y_true = torch.cat(target_chunks).cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": float(acc), "macro_f1": float(macro)}


def train_one_epoch_single(model, loader, criterion, optimizer, scaler, device, use_cuda_amp, use_mps_amp, pin_mem):
    model.train()
    loss_sum, total = 0.0, 0
    pred_chunks, target_chunks = [], []
    skipped_zero, coerced_multi = 0, 0

    for X, y_multi in loader:
        X = X.to(device, non_blocking=pin_mem)
        y_multi = y_multi.to(device, non_blocking=pin_mem)

        y_idx, valid_mask, multi_pos_mask = _single_targets_from_multi_hot(y_multi)
        skipped_zero += int((~valid_mask).sum().item())
        coerced_multi += int(multi_pos_mask.sum().item())
        if not bool(valid_mask.any()):
            continue

        optimizer.zero_grad(set_to_none=True)

        context = torch.amp.autocast("cuda") if use_cuda_amp else \
            (torch.autocast(device_type="mps", dtype=torch.float16) if use_mps_amp else torch.enable_grad())

        with context:
            logits = model(X)
            loss = criterion(logits[valid_mask], y_idx[valid_mask])

        if use_cuda_amp:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        pred = torch.argmax(logits[valid_mask].detach(), dim=1).cpu()
        tgt = y_idx[valid_mask].detach().cpu()
        pred_chunks.append(pred)
        target_chunks.append(tgt)
        loss_sum += float(loss.item()) * int(valid_mask.sum().item())
        total += int(valid_mask.sum().item())

    metrics = _compute_single_metrics(pred_chunks, target_chunks)
    return loss_sum / max(1, total), metrics, skipped_zero, coerced_multi


def evaluate_single(model, loader, criterion, device, use_cuda_amp, use_mps_amp, pin_mem):
    model.eval()
    loss_sum, total = 0.0, 0
    pred_chunks, target_chunks = [], []
    skipped_zero, coerced_multi = 0, 0

    with torch.no_grad():
        for X, y_multi in loader:
            X = X.to(device, non_blocking=pin_mem)
            y_multi = y_multi.to(device, non_blocking=pin_mem)

            y_idx, valid_mask, multi_pos_mask = _single_targets_from_multi_hot(y_multi)
            skipped_zero += int((~valid_mask).sum().item())
            coerced_multi += int(multi_pos_mask.sum().item())
            if not bool(valid_mask.any()):
                continue

            context = torch.amp.autocast("cuda") if use_cuda_amp else \
                (torch.autocast(device_type="mps", dtype=torch.float16) if use_mps_amp else torch.no_grad())

            with context:
                logits = model(X)
                loss = criterion(logits[valid_mask], y_idx[valid_mask])

            pred = torch.argmax(logits[valid_mask].detach(), dim=1).cpu()
            tgt = y_idx[valid_mask].detach().cpu()
            pred_chunks.append(pred)
            target_chunks.append(tgt)
            loss_sum += float(loss.item()) * int(valid_mask.sum().item())
            total += int(valid_mask.sum().item())

    metrics = _compute_single_metrics(pred_chunks, target_chunks)
    return loss_sum / max(1, total), metrics, skipped_zero, coerced_multi


def single_label_train_loop(
    *,
    dataset,
    classes: List[str],
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
    feature_mode: str,
    resume_from: Optional[Path] = None,
    backbone: str = "densenet121",
):
    seed_everything(seed)
    device, use_cuda_amp, use_mps_amp, scaler, pin_mem = get_device()

    val_size = int(round(len(dataset) * val_frac))
    train_ds, val_ds = random_split(
        dataset,
        [len(dataset) - val_size, val_size],
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

    inferred_in_ch, inferred_freq_bins = infer_input_shape(dataset)
    model_in_ch = inferred_in_ch
    model_freq_bins = inferred_freq_bins

    model_backbone = str(backbone).strip().lower()
    if resume_from and resume_from.exists():
        try:
            ckpt_meta = torch.load(resume_from, map_location="cpu")
            ckpt_in_ch = ckpt_meta.get("in_ch")
            ckpt_freq_bins = ckpt_meta.get("freq_bins")
            ckpt_backbone = str(ckpt_meta.get("backbone", model_backbone)).strip().lower()
            if ckpt_in_ch is not None and ckpt_in_ch != model_in_ch:
                print(f"[WARN] Checkpoint in_ch={ckpt_in_ch} differs from dataset in_ch={model_in_ch}; using checkpoint.")
                model_in_ch = ckpt_in_ch
            if ckpt_freq_bins is not None and ckpt_freq_bins != model_freq_bins:
                print(f"[WARN] Checkpoint freq_bins={ckpt_freq_bins} differs from dataset freq_bins={model_freq_bins}; using checkpoint.")
                model_freq_bins = ckpt_freq_bins
            if ckpt_backbone != model_backbone:
                print(f"[WARN] Checkpoint backbone={ckpt_backbone} differs from requested backbone={model_backbone}; using checkpoint.")
                model_backbone = ckpt_backbone
        except Exception as exc:
            print(f"[WARN] Failed to read checkpoint metadata: {exc}")

    model = build_model(
        num_classes=len(classes),
        dropout=dropout,
        in_ch=model_in_ch,
        freq_bins=model_freq_bins,
        device=device,
        backbone=model_backbone,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc", "train_macro_f1", "val_macro_f1"]}
    start_epoch, best_val_macro, no_improve = 1, 0.0, 0

    if resume_from and resume_from.exists():
        ckpt = load_checkpoint(resume_from, device, model, optimizer, scheduler, scaler)
        start_epoch = ckpt["epoch"] + 1
        history = ckpt["history"]
        best_val_macro = ckpt.get("best_val_macro_f1", 0.0)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        train_loss, train_m, train_zero, train_multi = train_one_epoch_single(
            model, train_loader, criterion, optimizer, scaler, device, use_cuda_amp, use_mps_amp, pin_mem
        )
        val_loss, val_m, val_zero, val_multi = evaluate_single(
            model, val_loader, criterion, device, use_cuda_amp, use_mps_amp, pin_mem
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_m["accuracy"])
        history["val_acc"].append(val_m["accuracy"])
        history["train_macro_f1"].append(train_m["macro_f1"])
        history["val_macro_f1"].append(val_m["macro_f1"])

        print(
            f"[{epoch}/{epochs}] Loss: {train_loss:.4f}/{val_loss:.4f} | "
            f"Val Acc: {val_m['accuracy']:.4f} | Val MacroF1: {val_m['macro_f1']:.4f} | "
            f"Time: {time.time()-t0:.1f}s"
        )
        if train_zero or val_zero:
            print(f"[INFO] Single-label mode skipped zero-label samples (train={train_zero}, val={val_zero})")
        if train_multi or val_multi:
            print(f"[INFO] Single-label mode coerced multi-positive samples via argmax (train={train_multi}, val={val_multi})")

        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "opt_state": optimizer.state_dict(),
            "history": history,
            "best_val_macro_f1": best_val_macro,
            "classes": classes,
            "audio_config": audio_cfg,
            "label_to_idx": getattr(dataset, "label_to_idx", {name: i for i, name in enumerate(classes)}),
            "feature_mode": feature_mode,
            "task_mode": "single_label",
            "in_ch": model_in_ch,
            "freq_bins": model_freq_bins,
            "backbone": model_backbone,
        }
        save_checkpoint(payload, ckpt_dir / "last.pt")

        if val_m["macro_f1"] > best_val_macro:
            best_val_macro = val_m["macro_f1"]
            no_improve = 0
            save_checkpoint(payload, ckpt_dir / "best_val.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return {"model": model, "history": history, "classes": classes, "audio_config": audio_cfg}
