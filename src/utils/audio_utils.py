import yaml
import math
from pathlib import Path
from typing import Optional, Set, Sequence

def parse_dataset(dataset_dir: Path):
    """Scans a directory for .wav files, treating the parent folder as the label."""
    for wav_path in dataset_dir.rglob("*.wav"):
        if wav_path.is_file():
            yield wav_path, wav_path.parent.name.strip().lower()

def max_safe_cqt_bins(sr: int, fmin: float, bins_per_octave: int) -> int:
    """Returns the largest valid CQT bin count that keeps top frequency <= Nyquist."""
    nyquist = sr / 2.0
    if fmin >= nyquist:
        return 1
    return int(math.floor(bins_per_octave * math.log2(nyquist / fmin))) + 1

def load_allowed_labels(labels_config: Optional[Path], dataset_key: str, label_key: Optional[str]) -> Optional[Set[str]]:
    """Extracts target labels from the labels.yaml configuration."""
    if not labels_config or not labels_config.exists():
        return None

    with open(labels_config, "r", encoding="utf-8") as f:
        labels_cfg = yaml.safe_load(f) or {}

    keys_to_try = [label_key] if label_key else []
    keys_to_try.extend(["irmas_labels", "train_labels"] if dataset_key == "irmas" else ["train_labels", "labels"])

    for key in keys_to_try:
        maybe = labels_cfg.get(key)
        if isinstance(maybe, (list, tuple)):
            return {str(x).strip().lower() for x in maybe if x is not None}
            
    return None