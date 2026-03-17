# src/data_loader.py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset

from src.feature_modes import align_and_stack_feature_tensors, feature_mode_to_features, normalize_feature_mode

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
        feature_mode: str,
        manifest_path: str | Path,
        class_names: list,
        cqt_manifest_path: Optional[str | Path] = None,
        feature_manifest_paths: Optional[dict[str, str | Path]] = None,
        project_root: str = ".",
        transform=None,
        max_zero_label_warnings: int = 10,
        infer_label_from_parent: bool = True,
    ):
        self.feature_mode = normalize_feature_mode(feature_mode)
        self.feature_names = feature_mode_to_features(self.feature_mode)
        self.root = Path(project_root)
        self.transform = transform
        self.infer_label_from_parent = infer_label_from_parent
        
        self.class_names = [c.strip().lower() for c in class_names]
        self.label_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        manifest_paths: dict[str, str | Path] = dict(feature_manifest_paths or {})
        if not manifest_paths:
            manifest_paths[self.feature_names[0]] = manifest_path
            if "cqt" in self.feature_names and cqt_manifest_path is not None:
                manifest_paths["cqt"] = cqt_manifest_path

        missing_features = [name for name in self.feature_names if name not in manifest_paths]
        if missing_features:
            raise ValueError(
                f"Missing manifest paths for feature_mode='{self.feature_mode}': {', '.join(missing_features)}"
            )

        # 1. Load and align DataFrames for the requested feature set.
        df = self._load_feature_dataframe(self.feature_names[0], manifest_paths[self.feature_names[0]])
        for feature_name in self.feature_names[1:]:
            df_feature = self._load_feature_dataframe(feature_name, manifest_paths[feature_name])
            df = pd.merge(df, df_feature, on=["wavpath", "label"], how="inner")

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

    @staticmethod
    def _load_feature_dataframe(feature_name: str, manifest_path: str | Path) -> pd.DataFrame:
        df = _read_csv_with_fallback(Path(manifest_path))
        if "filepath" not in df.columns:
            if feature_name == "cqt" and "cqt_path" in df.columns:
                df["filepath"] = df["cqt_path"]
            else:
                raise ValueError(f"Manifest for feature '{feature_name}' is missing a 'filepath' column.")

        if "label" not in df.columns and "labels" in df.columns:
            df["label"] = df["labels"]
        if "wavpath" not in df.columns and "sources" in df.columns:
            df["wavpath"] = df["sources"]

        required_cols = {"filepath", "label", "wavpath"}
        missing_cols = sorted(required_cols.difference(df.columns))
        if missing_cols:
            raise ValueError(
                f"Manifest for feature '{feature_name}' is missing required columns: {', '.join(missing_cols)}"
            )

        return df.rename(columns={"filepath": f"filepath_{feature_name}"})

    def _load_npy(self, path_str: str) -> torch.Tensor:
        npy_path = Path(path_str) if Path(path_str).is_absolute() else self.root / path_str
        return torch.from_numpy(np.load(npy_path)).float()

    def __getitem__(self, idx):
        self._seen += 1
        row = self.df.iloc[idx]

        feature_tensors = [self._load_npy(row[f"filepath_{name}"]) for name in self.feature_names]
        x = align_and_stack_feature_tensors(feature_tensors)

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
