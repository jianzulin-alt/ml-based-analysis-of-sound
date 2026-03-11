# src/train/utils.py
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

def seed_everything(seed: int = 1337):
    """Sets seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """Detects the best available device (CUDA, MPS, or CPU) and AMP support."""
    use_cuda_amp = False
    use_mps_amp = False
    pin_mem = False
    scaler = None

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

def get_autocast_context(use_cuda_amp: bool, use_mps_amp: bool):
    if use_cuda_amp:
        return torch.amp.autocast(device_type="cuda")
    if use_mps_amp:
        return torch.autocast(device_type="mps", dtype=torch.float16)
    return torch.enable_grad() # Null context fallback

def collate_fn_padd(batch):
    """Pads variable length spectrograms to the maximum width in the batch."""
    tensors = [item[0] for item in batch]
    targets = torch.stack([item[1] for item in batch])

    tensors = [t.permute(2, 0, 1) for t in tensors]
    tensors_padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    tensors_padded = tensors_padded.permute(0, 2, 3, 1)

    return tensors_padded, targets

def save_checkpoint(payload: Dict[str, Any], filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, filepath)

def load_checkpoint(path: Path, device: str, model: nn.Module, optimizer: torch.optim.Optimizer = None, scheduler: Any = None, scaler: Any = None):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    
    if optimizer and "opt_state" in ckpt:
        optimizer.load_state_dict(ckpt["opt_state"])
    if scheduler and "sched_state" in ckpt:
        scheduler.load_state_dict(ckpt["sched_state"])
    if scaler and ckpt.get("scaler_state") and scaler.is_enabled():
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt
