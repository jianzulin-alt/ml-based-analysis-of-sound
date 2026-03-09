import csv
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

import src

PROJECT_ROOT = Path(src.__file__).resolve().parents[1]
SUPPORTED_MODELS = ("cnn", "crnn", "mobilenet_v3_small")


def make_cuda_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def get_autocast_context(use_cuda_amp: bool, use_mps_amp: bool):
    if use_cuda_amp:
        try:
            return torch.amp.autocast(device_type="cuda")
        except (AttributeError, TypeError):
            return torch.cuda.amp.autocast()
    if use_mps_amp:
        return torch.autocast(device_type="mps", dtype=torch.float16)
    return nullcontext()


def seed_everything(seed: int = 1337):
    """Sets seeds for reproducibility across numpy, random, and torch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Detects the best available device (CUDA, MPS, or CPU)."""
    use_cuda_amp = False
    use_mps_amp = False
    scaler = make_cuda_grad_scaler(enabled=False)
    pin_mem = False

    if torch.cuda.is_available():
        device = "cuda"
        use_cuda_amp = True
        scaler = make_cuda_grad_scaler(enabled=True)
        pin_mem = True
    elif torch.backends.mps.is_available():
        device = "mps"
        use_mps_amp = True
    else:
        device = "cpu"

    return device, use_cuda_amp, use_mps_amp, scaler, pin_mem


def build_model(
    model_name: str,
    num_classes: int,
    dropout: float,
    in_ch: int = 2,
    pretrained: bool = False,
    device: str = "cpu",
):
    """Instantiates the requested model architecture and moves it to device."""
    model_key = model_name.strip().lower()

    if model_key == "cnn":
        from src.models.CNN import CNN

        model = CNN(in_ch=in_ch, num_classes=num_classes, p_drop=dropout)
    elif model_key == "crnn":
        from src.models.CRNN import CRNN

        model = CRNN(num_classes=num_classes, p_drop=dropout)
    elif model_key in {"mobilenetv3", "mobilenet_v3", "mobilenet_v3_small", "mobilenetv3_small"}:
        from src.models.MobileNetV3 import MobileNetV3Small

        model = MobileNetV3Small(
            num_classes=num_classes,
            in_ch=in_ch,
            dropout=dropout,
            pretrained=pretrained,
        )
    else:
        supported = ", ".join(SUPPORTED_MODELS)
        raise ValueError(f"Unsupported model '{model_name}'. Supported: {supported}")

    return model.to(device)


def set_feature_extractor_trainable(model: nn.Module, trainable: bool) -> bool:
    """
    Freezes/unfreezes feature extractor when model exposes hooks.
    Returns False when operation is not supported for the model.
    """
    has_hooks = hasattr(model, "freeze_feature_extractor") and hasattr(model, "unfreeze_feature_extractor")
    if not has_hooks:
        return False

    if trainable:
        model.unfreeze_feature_extractor()
    else:
        model.freeze_feature_extractor()
    return True


def save_checkpoint(payload: Dict[str, Any], filepath: Path):
    """Saves the training state to a file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, filepath)


def load_model_weights(path: Path, device: str, model: nn.Module):
    """Loads model weights only (warm start), without optimiser/scheduler state."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return ckpt


def load_checkpoint(
    path: Path,
    device: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    current_label_to_idx: Optional[Dict[str, int]] = None,
):
    """Loads a saved checkpoint and restores model + optimizer state."""
    ckpt = torch.load(path, map_location=device)

    if current_label_to_idx and "label_to_idx" in ckpt:
        if len(current_label_to_idx) != len(ckpt["label_to_idx"]):
            print("Warning: Number of classes in checkpoint doesn't match current config!")

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["opt_state"])

    if scheduler and "sched_state" in ckpt:
        scheduler.load_state_dict(ckpt["sched_state"])

    if scaler and ckpt.get("scaler_state") and scaler.is_enabled():
        scaler.load_state_dict(ckpt["scaler_state"])

    return ckpt


def collate_fn_padd(batch):
    """Pads variable length mel spectrograms to the maximum width in the batch."""
    tensors = [item[0] for item in batch]
    targets = torch.stack([item[1] for item in batch])

    tensors = [t.permute(2, 0, 1) for t in tensors]
    tensors_padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    tensors_padded = tensors_padded.permute(0, 2, 3, 1)

    return tensors_padded, targets


def write_history_csv(history: Dict[str, List[float]], filepath: Path) -> None:
    metric_keys = list(history.keys())
    if not metric_keys:
        return

    epochs = len(history[metric_keys[0]])
    for key in metric_keys:
        if len(history[key]) != epochs:
            raise ValueError(f"Inconsistent history lengths. Metric '{key}' has length {len(history[key])}, expected {epochs}.")

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", *metric_keys])
        for i in range(epochs):
            writer.writerow([i + 1, *[history[k][i] for k in metric_keys]])


def append_experiment_log(filepath: Path, row: Dict[str, Any], fieldnames: Sequence[str]) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    file_exists = filepath.exists()
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})
