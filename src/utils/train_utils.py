import csv
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import os
import time
import random
import contextlib
from typing import Tuple, Optional, Any, Dict, List
import numpy as np

def seed_everything(seed: int = 1337) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> Tuple[str, bool, bool, Optional[torch.amp.GradScaler], bool]:
    use_cuda_amp, use_mps_amp, pin_mem = False, False, False
    scaler = None
    if torch.cuda.is_available():
        device, use_cuda_amp, pin_mem = "cuda", True, True
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    elif torch.backends.mps.is_available():
        device, use_mps_amp = "mps", True
    else:
        device = "cpu"
    return device, use_cuda_amp, use_mps_amp, scaler, pin_mem

def get_autocast_context(use_cuda_amp: bool, use_mps_amp: bool) -> Any:
    if use_cuda_amp: return torch.amp.autocast(device_type="cuda")
    if use_mps_amp: return torch.amp.autocast(device_type="mps", dtype=torch.float16)
    return contextlib.nullcontext()

def collate_fn_padd(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    tensors = [item[0].permute(2, 0, 1) for item in batch]
    targets = torch.stack([item[1] for item in batch])
    tensors_padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    return tensors_padded.permute(0, 2, 3, 1), targets

def save_checkpoint(payload: Dict[str, Any], filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = filepath.with_name(f"{filepath.name}.tmp-{os.getpid()}")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, filepath)

def load_checkpoint(
    path: Path, device: str, model: nn.Module, 
    optimizer: Optional[torch.optim.Optimizer] = None, 
    scheduler: Any = None, scaler: Any = None
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "opt_state" in ckpt: optimizer.load_state_dict(ckpt["opt_state"])
    if scheduler and "sched_state" in ckpt: scheduler.load_state_dict(ckpt["sched_state"])
    if scaler and ckpt.get("scaler_state") and scaler.is_enabled(): scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt

def prepare_manifest_for_dataset(
    manifest_path: Path, run_dir: Path, ensure_filepath_from_cqt: bool, ensure_merge_keys: bool
) -> Path:
    """Normalises manifest columns so FeatureFusionDataset can consume diverse CSV formats."""
    df = pd.read_csv(manifest_path)
    changed = False

    if ensure_filepath_from_cqt and "filepath" not in df.columns and "cqt_path" in df.columns:
        df["filepath"] = df["cqt_path"]
        changed = True

    if ensure_merge_keys:
        if "label" not in df.columns and "labels" in df.columns:
            df["label"] = df["labels"]
            changed = True
        if "wavpath" not in df.columns and "sources" in df.columns:
            df["wavpath"] = df["sources"]
            changed = True

    if not changed:
        return manifest_path

    out_dir = run_dir / "tmp_manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{manifest_path.stem}__adapted.csv"
    df.to_csv(out_path, index=False)
    return out_path

def count_model_parameters(model: nn.Module) -> Tuple[int, int]:
    """Returns (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def write_history_csv(history: dict, out_path: Path) -> None:
    """Writes a training history dictionary to a CSV file."""
    keys = sorted(history.keys())
    n_rows = max((len(v) for v in history.values() if isinstance(v, list)), default=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(n_rows):
            row = [history.get(k, [])[i] if i < len(history.get(k, [])) else "" for k in keys]
            writer.writerow(row)