# src/data_loader.py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset

def _read_csv_with_fallback(path: Path) -> pd.DataFrame:
    encodings = ("utf-8", "utf-8-sig", "gbk", "cp936")
    last_err: Optional[UnicodeDecodeError] = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
    if last_err is not None:
        raise ValueError(f"Failed to decode CSV: {path} with encodings {encodings}") from last_err
    return pd.read_csv(path)

class UniversalAudioDataset(Dataset):
    def __init__(
        self,
        feature_mode: str,          # 'mel', 'cqt', or 'mel_cqt'
        manifest_path: str | Path,  # Path to the Mel CSV (or CQT CSV if cqt mode)
        class_names: list,
        cqt_manifest_path: Optional[str | Path] = None, # Only required for 'mel_cqt'
        project_root: str = ".",
        transform=None,
        max_zero_label_warnings: int = 10,
        infer_label_from_parent: bool = True,
    ):
        self.feature_mode = str(feature_mode).strip().lower()
        self.root = Path(project_root)
        self.transform = transform
        self.infer_label_from_parent = infer_label_from_parent
        
        self.class_names = [c.strip().lower() for c in class_names]
        self.label_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        # 1. Load DataFrames
        df = _read_csv_with_fallback(Path(manifest_path))
        
        if self.feature_mode == "mel_cqt":
            if not cqt_manifest_path:
                raise ValueError("cqt_manifest_path must be provided for mel_cqt mode.")
            df_cqt = _read_csv_with_fallback(Path(cqt_manifest_path))
            # Merge on wavpath and label to ensure strict alignment
            df = pd.merge(df, df_cqt, on=["wavpath", "label"], suffixes=("_mel", "_cqt"))

        # 2. Canonicalise labels
        labels_series = df.get("labels", df.get("label"))
        if labels_series is None:
            raise ValueError("Manifest must contain a 'label' or 'labels' column")

        df["labels_raw"] = labels_series.fillna("").astype(str).str.lower().str.replace(" ", "", regex=False)
        df["labels_raw"] = df["labels_raw"].str.replace(",", "|", regex=False)

        self.df = df
        self._zero_label_count = 0
        self._seen = 0
        self._max_zero_label_warnings = max_zero_label_warnings

    def __len__(self):
        return len(self.df)

    def _load_npy(self, path_str: str) -> torch.Tensor:
        npy_path = Path(path_str) if Path(path_str).is_absolute() else self.root / path_str
        return torch.from_numpy(np.load(npy_path)).float()

    def __getitem__(self, idx):
        self._seen += 1
        row = self.df.iloc[idx]

        # 3. Load Features based on mode
        if self.feature_mode == "mel":
            x = self._load_npy(row["filepath"])
        elif self.feature_mode == "cqt":
            x = self._load_npy(row["filepath"])
        elif self.feature_mode == "mel_cqt":
            mel_tensor = self._load_npy(row["filepath_mel"])
            cqt_tensor = self._load_npy(row["filepath_cqt"])
            
            # Align time dimensions (crop to common size)
            min_w = min(mel_tensor.shape[2], cqt_tensor.shape[2])
            mel_tensor = mel_tensor[:, :, :min_w]
            cqt_tensor = cqt_tensor[:, :, :min_w]
            x = torch.cat([mel_tensor, cqt_tensor], dim=0)
        else:
            raise ValueError(f"Unknown feature mode: {self.feature_mode}")

        if self.transform is not None:
            x = self.transform(x)

        # 4. Parse Multi-hot Target
        raw = row["labels_raw"]
        label_list = [lbl for lbl in raw.split("|") if lbl]

        if not label_list and self.infer_label_from_parent:
            inferred = Path(str(row.get("wavpath", ""))).parent.name.strip().lower()
            if inferred in self.label_to_idx:
                label_list = [inferred]

        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for label in label_list:
            if label in self.label_to_idx:
                target[self.label_to_idx[label]] = 1.0

        if target.sum().item() == 0 and self._zero_label_count <= self._max_zero_label_warnings:
            self._zero_label_count += 1
            print(f"[WARN] All-zero target produced at idx={idx}, raw labels='{raw}'")

        return x, target