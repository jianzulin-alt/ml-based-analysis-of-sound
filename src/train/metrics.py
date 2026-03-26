from __future__ import annotations
from typing import Dict
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

def _safe_metric(value: float) -> float:
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return float(value)

def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, task_mode: str = "single_label") -> Dict[str, float]:
    """Computes metrics dynamically based on the task mode (Single vs Multi Label)."""
    if targets.numel() == 0:
        return {"acc": 0.0, "macro_f1": 0.0, "micro_f1": 0.0}

    # 1. Single Label Logic (Softmax / Argmax)
    if task_mode == "single_label":
        if targets.ndim > 1 and targets.shape[1] > 1:
            y_true = torch.argmax(targets, dim=1).cpu().numpy()
        else:
            y_true = targets.cpu().numpy()
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()

    # 2. Multi-Label Logic (Sigmoid / 0.5 Threshold)
    elif task_mode == "multi_label":
        y_true = targets.cpu().numpy()
        probs = torch.sigmoid(logits)
        y_pred = (probs > 0.5).int().cpu().numpy()
        
    else:
        raise ValueError(f"Unknown task_mode: {task_mode}")

    # 3. Calculate Metrics
    # For multi-label, exact match accuracy is very strict. F1-macro is your primary target.
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    
    return {
        "acc": _safe_metric(acc),
        "macro_f1": _safe_metric(macro_f1),
        "micro_f1": _safe_metric(micro_f1),
    }