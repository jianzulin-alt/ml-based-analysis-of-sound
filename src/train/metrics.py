from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def _safe_metric(value: float) -> float:
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return float(value)


def compute_single_label_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute single-label classification metrics.
    """
    if y_true.size == 0:
        return {"acc": 0.0, "macro_f1": 0.0, "micro_f1": 0.0}

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    return {
        "acc": _safe_metric(acc),
        "macro_f1": _safe_metric(macro_f1),
        "micro_f1": _safe_metric(micro_f1),
    }


"""
def compute_multi_label_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    # Compute multi-label classification metrics from probabilities.
 
    if y_true.size == 0:
        return {
            "hamming_acc": 0.0,
            "subset_acc": 0.0,
            "macro_f1": 0.0,
            "micro_f1": 0.0,
        }

    y_pred = (y_prob >= float(threshold)).astype(np.int32)
    # Exact match ratio (subset accuracy).
    subset_acc = np.mean(np.all(y_pred == y_true, axis=1))
    # Per-label elementwise accuracy (1 - hamming loss).
    hamming_acc = np.mean(y_pred == y_true)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    return {
        "hamming_acc": _safe_metric(hamming_acc),
        "subset_acc": _safe_metric(subset_acc),
        "macro_f1": _safe_metric(macro_f1),
        "micro_f1": _safe_metric(micro_f1),
    }
"""