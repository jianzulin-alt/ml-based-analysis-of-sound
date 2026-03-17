from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.feature_modes import (
    align_and_stack_feature_tensors,
    feature_mode_to_features,
    feature_mode_to_in_channels,
    normalize_feature_mode,
)
from src.models.CNN import CNN
from src.models.CNN_DenseNet_121 import CNN_DenseNet_121
from src.models.CNN_MultiFeatureFusionAttention import (
    BaselineMultiFeatureCNN,
    ModelConfig,
    MultiFeatureFusionAttentionCNNLogits,
)
from src.preprocessing.features import (
    compute_stereo_cqt_db,
    compute_stereo_logmel_db,
    compute_stereo_mfcc,
    compute_stereo_chroma,
    compute_stft_params,
)
from src.preprocessing.preprocessing import (
    conform_audio_duration,
    load_audio_as_stereo_and_resample,
    preprocess_loudness,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path_like: str | Path, root: Path) -> Path:
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p.resolve()
    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (root / p).resolve()


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def choose_classes(labels_cfg: dict, dataset_name: str) -> list[str]:
    keys = ["train_labels", "labels"]
    if dataset_name == "irmas":
        keys = ["irmas_labels", "train_labels", "labels"]

    for key in keys:
        labels = labels_cfg.get(key)
        if labels:
            classes = [str(x).strip().lower() for x in labels if x is not None]
            if classes:
                return classes

    raise ValueError(f"No labels found for dataset='{dataset_name}'. Tried keys: {keys}.")


def choose_classes_from_run_config(run_cfg: dict[str, Any]) -> list[str]:
    for key in ("classes", "class_names", "labels"):
        values = run_cfg.get(key)
        if values:
            classes = [str(x).strip().lower() for x in values if x is not None]
            if classes:
                return classes

    resolved_cfg = run_cfg.get("resolved", {}) if isinstance(run_cfg.get("resolved"), dict) else {}
    for key in ("classes", "class_names", "labels"):
        values = resolved_cfg.get(key)
        if values:
            classes = [str(x).strip().lower() for x in values if x is not None]
            if classes:
                return classes

    return []


def select_device(device: str = "auto") -> torch.device:
    raw = str(device).strip().lower()
    if raw == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(raw)


def build_model(backbone: str, in_ch: int, num_classes: int, model_cfg: dict) -> torch.nn.Module:
    name = str(backbone).strip().lower()
    dropout = float(model_cfg.get("dropout", 0.3))
    if name == "cnn":
        return CNN(in_ch=in_ch, num_classes=num_classes, p_drop=dropout)
    if name == "cnn_densenet_121":
        return CNN_DenseNet_121(in_ch=in_ch, num_classes=num_classes, p_drop=dropout)
    if name in {"baseline_multifeature_cnn", "cnn_multifeature_baseline"}:
        return BaselineMultiFeatureCNN(
            ModelConfig(
                in_channels=in_ch,
                num_classes=num_classes,
                fc_hidden_dim=int(model_cfg.get("fc_hidden_dim", 256)),
                attention_reduction=int(model_cfg.get("attention_reduction", 8)),
                dropout=dropout,
            )
        )
    if name in {"fusion_attention_cnn", "multi_feature_fusion_attention", "cnn_multifeature_fusion_attention"}:
        return MultiFeatureFusionAttentionCNNLogits(
            ModelConfig(
                in_channels=in_ch,
                num_classes=num_classes,
                fc_hidden_dim=int(model_cfg.get("fc_hidden_dim", 256)),
                attention_reduction=int(model_cfg.get("attention_reduction", 8)),
                dropout=dropout,
            )
        )
    raise ValueError(f"Unsupported backbone: {backbone}")


def _f1(precision: float, recall: float) -> float:
    return float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


@dataclass
class TestSample:
    wav_path: Path
    txt_path: Path
    all_labels: list[str]
    relevant_labels: list[str]


def _read_txt_labels(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    labels: list[str] = []
    for line in raw.splitlines():
        token = line.strip().lower().replace("\t", " ")
        token = token.split()[0] if token else ""
        if token:
            labels.append(token)
    return list(dict.fromkeys(labels))


def collect_test_samples(test_root: Path, valid_labels_set: set[str]) -> list[TestSample]:
    samples: list[TestSample] = []
    for wav_path in sorted(test_root.rglob("*.wav")):
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            continue

        all_labels = _read_txt_labels(txt_path)
        if not all_labels:
            continue

        relevant_labels = [label for label in all_labels if label in valid_labels_set]
        samples.append(
            TestSample(
                wav_path=wav_path,
                txt_path=txt_path,
                all_labels=all_labels,
                relevant_labels=relevant_labels,
            )
        )
    return samples


class OnTheFlyTestDataset(Dataset):
    def __init__(
        self,
        samples: list[TestSample],
        classes: list[str],
        audio_cfg: dict,
        feature_mode: str,
    ) -> None:
        self.samples = samples
        self.classes = [c.strip().lower() for c in classes]
        self.label_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.feature_mode = normalize_feature_mode(feature_mode)
        self.feature_names = feature_mode_to_features(self.feature_mode)

        self.sr = int(audio_cfg["sr"])
        self.duration = float(audio_cfg["duration"])
        self.n_fft, self.hop, self.win_length = compute_stft_params(
            self.sr,
            float(audio_cfg["win_ms"]),
            float(audio_cfg["hop_ms"]),
        )
        self.window = str(audio_cfg.get("window", "hann"))
        self.fmin = float(audio_cfg.get("fmin", 20.0))
        self.fmax = float(audio_cfg.get("fmax", self.sr / 2))
        self.n_mels = int(audio_cfg.get("n_mels", 128))
        self.n_bins = int(audio_cfg.get("n_bins", 120))
        self.bins_per_octave = int(audio_cfg.get("bins_per_octave", 12))
        self.n_mfcc = int(audio_cfg.get("n_mfcc", 13))
        self.n_chroma = int(audio_cfg.get("n_chroma", 12))
        self.loudness_norm = str(audio_cfg.get("loudness_norm", "none"))
        self.target_lufs = float(audio_cfg.get("target_lufs", -23.0))
        self.loudness_peak_limit = float(audio_cfg.get("loudness_peak_limit", 0.99))

    def __len__(self) -> int:
        return len(self.samples)

    def _extract_single_feature(self, stereo: np.ndarray, feature_name: str) -> torch.Tensor:
        if feature_name == "mel":
            return torch.from_numpy(
                compute_stereo_logmel_db(
                    stereo,
                    self.sr,
                    n_fft=self.n_fft,
                    hop=self.hop,
                    win_length=self.win_length,
                    n_mels=self.n_mels,
                    fmin=self.fmin,
                    fmax=self.fmax,
                    window=self.window,
                )
            ).float()
        if feature_name == "cqt":
            return torch.from_numpy(
                compute_stereo_cqt_db(
                    stereo,
                    self.sr,
                    n_bins=self.n_bins,
                    bins_per_octave=self.bins_per_octave,
                    hop_length=self.hop,
                    fmin=self.fmin,
                )
            ).float()
        if feature_name == "mfcc":
            return torch.from_numpy(
                compute_stereo_mfcc(
                    stereo,
                    self.sr,
                    n_fft=self.n_fft,
                    hop=self.hop,
                    win_length=self.win_length,
                    n_mfcc=self.n_mfcc,
                    n_mels=self.n_mels,
                    fmin=self.fmin,
                    fmax=self.fmax,
                    window=self.window,
                )
            ).float()
        if feature_name == "chroma":
            return torch.from_numpy(
                compute_stereo_chroma(
                    stereo,
                    self.sr,
                    n_fft=self.n_fft,
                    hop=self.hop,
                    win_length=self.win_length,
                    n_chroma=self.n_chroma,
                    fmin=self.fmin,
                    window=self.window,
                )
            ).float()
        raise ValueError(f"Unsupported feature name: {feature_name}")

    def _extract_features(self, wav_path: Path) -> torch.Tensor:
        stereo = load_audio_as_stereo_and_resample(wav_path, target_sr=self.sr)
        stereo = conform_audio_duration(stereo, self.sr, self.duration)
        stereo = preprocess_loudness(
            stereo,
            sr=self.sr,
            loudness_norm=self.loudness_norm,
            target_lufs=self.target_lufs,
            peak_limit=self.loudness_peak_limit,
        )

        feature_tensors = [self._extract_single_feature(stereo, name) for name in self.feature_names]
        return align_and_stack_feature_tensors(feature_tensors)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        x = self._extract_features(sample.wav_path)

        true_vec = torch.zeros(len(self.classes), dtype=torch.float32)
        for label in sample.relevant_labels:
            label_idx = self.label_to_idx.get(label)
            if label_idx is not None:
                true_vec[label_idx] = 1.0
        return x, int(idx), true_vec


def _resolve_model_artifacts(model_path: str | Path, root: Path) -> dict[str, Any]:
    resolved = resolve_path(model_path, root)
    if not resolved.exists():
        raise FileNotFoundError(f"Model path not found: {resolved}")

    if resolved.is_file():
        if resolved.suffix != ".pt":
            raise ValueError(f"Expected a .pt checkpoint file, got: {resolved}")
        ckpt_path = resolved
        run_dir = resolved.parent
    else:
        run_dir = resolved
        best = run_dir / "best_val.pt"
        last = run_dir / "last.pt"
        if best.exists():
            ckpt_path = best
        elif last.exists():
            ckpt_path = last
        else:
            raise FileNotFoundError(
                f"No checkpoint found in run dir: {run_dir}. Expected best_val.pt or last.pt."
            )

    return {
        "run_dir": run_dir,
        "checkpoint_path": ckpt_path,
        "run_config_path": run_dir / "run_config.yaml",
        "model_name": run_dir.name,
    }


def _load_run_config_with_fallback(run_config_path: Path, ckpt: dict[str, Any]) -> dict[str, Any]:
    if run_config_path.exists():
        try:
            cfg = load_yaml(run_config_path)
            if isinstance(cfg, dict) and cfg:
                return cfg
        except Exception:
            pass
    cfg = ckpt.get("config", {})
    return cfg if isinstance(cfg, dict) else {}


def _resolve_eval_config_paths(
    run_cfg: dict[str, Any],
    root: Path,
    audio_config_path: str | Path,
    labels_config_path: str | Path,
) -> tuple[Path, Path]:
    resolved_cfg = run_cfg.get("resolved", {}) if isinstance(run_cfg.get("resolved"), dict) else {}

    audio_cfg_path = resolve_path(
        resolved_cfg.get("audio_config", str(audio_config_path)),
        root,
    )
    labels_cfg_path = resolve_path(
        resolved_cfg.get("labels_config", str(labels_config_path)),
        root,
    )
    return audio_cfg_path, labels_cfg_path


def _resolve_test_root(
    dataset_name: str,
    datasets_cfg: dict,
    root: Path,
    test_root_override: Optional[str | Path] = None,
) -> Path:
    if test_root_override is not None and str(test_root_override).strip():
        return resolve_path(test_root_override, root)

    dataset_cfg = datasets_cfg.get(dataset_name, {}) if isinstance(datasets_cfg, dict) else {}
    test_rel = dataset_cfg.get("test", "")
    if test_rel:
        return resolve_path(test_rel, root)

    default_tests = {
        "irmas": "data/IRMAS/IRMAS-TestingData-Part1/Part1",
        "chinese_instruments": "data/test/a-touch-of-zen/",
    }
    default_rel = default_tests.get(dataset_name)
    if default_rel:
        return resolve_path(default_rel, root)

    raise ValueError(
        f"No test path configured for dataset='{dataset_name}'. "
        "Set datasets.<name>.test in audio config or pass test_root_override."
    )


def _build_prediction_row(
    sample: TestSample,
    classes: list[str],
    probs_row: np.ndarray,
    pred_idx: int,
    topk_idxs: list[int],
    top_k: int,
) -> dict[str, Any]:
    pred_label = classes[pred_idx]
    topk_labels = [classes[int(i)] for i in topk_idxs]

    all_label_set = set(sample.all_labels)
    relevant_label_set = set(sample.relevant_labels)
    has_relevant = bool(sample.relevant_labels)

    top1_hit_all = pred_label in all_label_set
    topk_hit_all = any(label in all_label_set for label in topk_labels)

    top1_hit_relevant = bool(pred_label in relevant_label_set) if has_relevant else np.nan
    topk_hit_relevant = (
        bool(any(label in relevant_label_set for label in topk_labels)) if has_relevant else np.nan
    )

    matched_topk_all = sorted(all_label_set.intersection(topk_labels))
    matched_topk_relevant = sorted(relevant_label_set.intersection(topk_labels))

    return {
        "wav": sample.wav_path.name,
        "all_true_labels": "|".join(sample.all_labels),
        "relevant_true_labels": "|".join(sample.relevant_labels),
        "pred_top1_label": pred_label,
        "pred_top1_conf": float(probs_row[pred_idx]),
        "pred_topk_labels": "|".join(topk_labels),
        "num_all_true_labels": int(len(sample.all_labels)),
        "num_relevant_true_labels": int(len(sample.relevant_labels)),
        "has_relevant_labels": bool(has_relevant),
        "top1_hit_all_labels": bool(top1_hit_all),
        "top1_hit_relevant_labels": top1_hit_relevant,
        f"top{top_k}_hit_all_labels": bool(topk_hit_all),
        f"top{top_k}_hit_relevant_labels": topk_hit_relevant,
        "matched_top1_all_label_count": int(top1_hit_all),
        "matched_top1_relevant_label_count": int(pred_label in relevant_label_set),
        f"matched_top{top_k}_all_label_count": int(len(matched_topk_all)),
        f"matched_top{top_k}_relevant_label_count": int(len(matched_topk_relevant)),
    }


def _build_per_class_df(pred_df: pd.DataFrame, classes: list[str], top_k: int) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame(
            columns=[
                "class_name",
                "n_occurrences",
                "top1_pickup_rate",
                f"top{top_k}_pickup_rate",
            ]
        )

    relevant_mask = pred_df["has_relevant_labels"] == True
    relevant_df = pred_df.loc[relevant_mask].copy()
    if relevant_df.empty:
        return pd.DataFrame(
            columns=[
                "class_name",
                "n_occurrences",
                "top1_pickup_rate",
                f"top{top_k}_pickup_rate",
            ]
        )

    rows: list[dict[str, Any]] = []
    for class_name in classes:
        mask = relevant_df["relevant_true_labels"].apply(
            lambda s, label=class_name: label in set(str(s).split("|")) if str(s) else False
        )
        n_occurrences = int(mask.sum())
        if n_occurrences == 0:
            continue

        class_rows = relevant_df.loc[mask]
        rows.append(
            {
                "class_name": class_name,
                "n_occurrences": n_occurrences,
                "top1_pickup_rate": float((class_rows["pred_top1_label"] == class_name).mean()),
                f"top{top_k}_pickup_rate": float(
                    class_rows["pred_topk_labels"].apply(
                        lambda s, label=class_name: label in set(str(s).split("|")) if str(s) else False
                    ).mean()
                ),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "class_name",
                "n_occurrences",
                "top1_pickup_rate",
                f"top{top_k}_pickup_rate",
            ]
        )

    return pd.DataFrame(rows).sort_values("class_name").reset_index(drop=True)


def _build_confusion_df(pred_df: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    confusion_df = pd.DataFrame(0, index=classes, columns=classes)
    if pred_df.empty:
        return confusion_df

    relevant_df = pred_df.loc[pred_df["has_relevant_labels"] == True].copy()
    for _, row in relevant_df.iterrows():
        pred_label = str(row["pred_top1_label"])
        true_labels = [label for label in str(row["relevant_true_labels"]).split("|") if label]
        for true_label in true_labels:
            if true_label in confusion_df.index and pred_label in confusion_df.columns:
                confusion_df.loc[true_label, pred_label] += 1
    return confusion_df


def _evaluate_run_impl(
    model_path: str | Path,
    *,
    dataset_fallback: str,
    audio_config_path: str | Path,
    labels_config_path: str | Path,
    test_root_override: Optional[str | Path],
    batch_size: int,
    num_workers: int,
    top_k: int,
    max_samples: Optional[int],
    device: str,
    show_progress: bool,
) -> dict[str, Any]:
    root = _repo_root()
    artifacts = _resolve_model_artifacts(model_path, root)
    ckpt = torch.load(artifacts["checkpoint_path"], map_location="cpu", weights_only=False)
    run_cfg = _load_run_config_with_fallback(artifacts["run_config_path"], ckpt)

    audio_cfg_path, labels_cfg_path = _resolve_eval_config_paths(
        run_cfg,
        root,
        audio_config_path,
        labels_config_path,
    )
    audio_cfg_all = load_yaml(audio_cfg_path)
    labels_cfg = load_yaml(labels_cfg_path) if labels_cfg_path.exists() else {}

    dataset_name = str(run_cfg.get("dataset", dataset_fallback)).strip().lower()
    feature_mode = normalize_feature_mode(run_cfg.get("feature_mode", "mel"))
    model_cfg = run_cfg.get("model", {}) if isinstance(run_cfg.get("model"), dict) else {}
    backbone = str(model_cfg.get("backbone", "cnn")).strip().lower()

    datasets_cfg = audio_cfg_all.get("datasets", {}) or {}
    if dataset_name not in datasets_cfg and dataset_name not in {"irmas", "chinese_instruments"}:
        raise ValueError(f"Dataset '{dataset_name}' not found in {audio_cfg_path}")

    audio_cfg = audio_cfg_all.get("audio", {}) or {}
    if not audio_cfg:
        raise ValueError(f"Missing 'audio' section in {audio_cfg_path}")

    test_root = _resolve_test_root(
        dataset_name,
        datasets_cfg,
        root,
        test_root_override=test_root_override,
    )
    if not test_root.exists():
        raise FileNotFoundError(f"Test root not found: {test_root}")

    classes = choose_classes_from_run_config(run_cfg)
    if not classes:
        classes = choose_classes(labels_cfg, dataset_name)
    samples = collect_test_samples(test_root, valid_labels_set=set(classes))
    if max_samples is not None:
        samples = samples[: max(0, int(max_samples))]
    if not samples:
        raise ValueError(f"No valid test samples found under: {test_root}")

    infer_device = select_device(device)
    in_ch = feature_mode_to_in_channels(feature_mode)
    model = build_model(backbone=backbone, in_ch=in_ch, num_classes=len(classes), model_cfg=model_cfg)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=True)
    model = model.to(infer_device)
    model.eval()

    ds = OnTheFlyTestDataset(samples=samples, classes=classes, audio_cfg=audio_cfg, feature_mode=feature_mode)
    dl = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(infer_device.type == "cuda"),
    )

    top_k = max(1, int(top_k))
    rows: list[dict[str, Any]] = []

    tp_total = 0
    fp_total = 0
    fn_total = 0
    tp_by_class = np.zeros(len(classes), dtype=np.int64)
    fp_by_class = np.zeros(len(classes), dtype=np.int64)
    fn_by_class = np.zeros(len(classes), dtype=np.int64)

    iterator: Iterable = dl
    if show_progress:
        iterator = tqdm(dl, total=len(dl), desc=f"Eval {dataset_name}", leave=False)

    with torch.no_grad():
        for x, sample_indices, true_vecs in iterator:
            x = x.to(infer_device, non_blocking=(infer_device.type == "cuda"))
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu()

            pred_top1 = torch.argmax(probs, dim=1)
            pred_topk = torch.topk(probs, k=min(top_k, probs.size(1)), dim=1).indices

            sample_idx_np = sample_indices.cpu().numpy().tolist()
            pred_top1_np = pred_top1.cpu().numpy().tolist()
            pred_topk_np = pred_topk.cpu().numpy()
            probs_np = probs.numpy()
            true_vec_np = true_vecs.cpu().numpy()

            for i, sample_idx in enumerate(sample_idx_np):
                sample = samples[sample_idx]
                pred_idx = int(pred_top1_np[i])
                topk_idxs = pred_topk_np[i].tolist()
                relevant_true_indices = np.flatnonzero(true_vec_np[i] > 0.0).tolist()

                if relevant_true_indices:
                    if pred_idx in relevant_true_indices:
                        tp_total += 1
                        tp_by_class[pred_idx] += 1
                    else:
                        fp_total += 1
                        fp_by_class[pred_idx] += 1

                    fn_count = len(relevant_true_indices) - (1 if pred_idx in relevant_true_indices else 0)
                    fn_total += fn_count
                    for true_idx in relevant_true_indices:
                        if true_idx != pred_idx:
                            fn_by_class[true_idx] += 1

                rows.append(
                    _build_prediction_row(
                        sample=sample,
                        classes=classes,
                        probs_row=probs_np[i],
                        pred_idx=pred_idx,
                        topk_idxs=topk_idxs,
                        top_k=top_k,
                    )
                )

    pred_df = pd.DataFrame(rows)

    relevant_mask = pred_df["has_relevant_labels"] == True if not pred_df.empty else pd.Series([], dtype=bool)
    total_samples = int(len(pred_df))
    relevant_eval_samples = int(relevant_mask.sum()) if not pred_df.empty else 0
    relevant_missing_samples = total_samples - relevant_eval_samples

    micro_precision = float(tp_total / (tp_total + fp_total)) if (tp_total + fp_total) > 0 else 0.0
    micro_recall = float(tp_total / (tp_total + fn_total)) if (tp_total + fn_total) > 0 else 0.0
    micro_f1 = _f1(micro_precision, micro_recall)

    class_f1_vals: list[float] = []
    for class_idx in range(len(classes)):
        precision_denom = int(tp_by_class[class_idx] + fp_by_class[class_idx])
        recall_denom = int(tp_by_class[class_idx] + fn_by_class[class_idx])
        precision = float(tp_by_class[class_idx] / precision_denom) if precision_denom > 0 else 0.0
        recall = float(tp_by_class[class_idx] / recall_denom) if recall_denom > 0 else 0.0
        class_f1_vals.append(_f1(precision, recall))
    macro_f1 = float(np.mean(class_f1_vals)) if class_f1_vals else 0.0

    topk_all_col = f"top{top_k}_hit_all_labels"
    topk_relevant_col = f"top{top_k}_hit_relevant_labels"

    all_top1_acc = float(pred_df["top1_hit_all_labels"].mean()) if not pred_df.empty else 0.0
    all_topk_acc = float(pred_df[topk_all_col].mean()) if not pred_df.empty else 0.0

    relevant_top1_acc = (
        float(pd.to_numeric(pred_df.loc[relevant_mask, "top1_hit_relevant_labels"], errors="coerce").mean())
        if relevant_eval_samples > 0
        else float("nan")
    )
    relevant_topk_acc = (
        float(pd.to_numeric(pred_df.loc[relevant_mask, topk_relevant_col], errors="coerce").mean())
        if relevant_eval_samples > 0
        else float("nan")
    )

    total_all_true_labels = int(pred_df["num_all_true_labels"].sum()) if not pred_df.empty else 0
    total_relevant_true_labels = int(pred_df["num_relevant_true_labels"].sum()) if not pred_df.empty else 0

    matched_top1_all_total = int(pred_df["matched_top1_all_label_count"].sum()) if not pred_df.empty else 0
    matched_topk_all_total = int(pred_df[f"matched_top{top_k}_all_label_count"].sum()) if not pred_df.empty else 0
    matched_top1_relevant_total = (
        int(pred_df["matched_top1_relevant_label_count"].sum()) if not pred_df.empty else 0
    )
    matched_topk_relevant_total = (
        int(pred_df[f"matched_top{top_k}_relevant_label_count"].sum()) if not pred_df.empty else 0
    )

    label_pickup_acc_top1_all = (
        float(matched_top1_all_total / total_all_true_labels) if total_all_true_labels > 0 else 0.0
    )
    label_pickup_acc_topk_all = (
        float(matched_topk_all_total / total_all_true_labels) if total_all_true_labels > 0 else 0.0
    )
    label_pickup_acc_top1_relevant = (
        float(matched_top1_relevant_total / total_relevant_true_labels)
        if total_relevant_true_labels > 0
        else float("nan")
    )
    label_pickup_acc_topk_relevant = (
        float(matched_topk_relevant_total / total_relevant_true_labels)
        if total_relevant_true_labels > 0
        else float("nan")
    )

    unique_all_labels = sorted(
        {
            label
            for value in pred_df.get("all_true_labels", pd.Series([], dtype=str)).tolist()
            for label in str(value).split("|")
            if label
        }
    )
    unique_relevant_labels = sorted(
        {
            label
            for value in pred_df.get("relevant_true_labels", pd.Series([], dtype=str)).tolist()
            for label in str(value).split("|")
            if label
        }
    )

    summary = {
        "model_name": artifacts["model_name"],
        "checkpoint_path": str(artifacts["checkpoint_path"]),
        "run_config_path": str(artifacts["run_config_path"]),
        "dataset": dataset_name,
        "feature_mode": feature_mode,
        "backbone": backbone,
        "test_root": str(test_root),
        "device": str(infer_device),
        "samples": total_samples,
        "classes": int(len(classes)),
        "unique_all_labels_in_test": int(len(unique_all_labels)),
        "unique_relevant_labels_in_test": int(len(unique_relevant_labels)),
        "samples_with_relevant_labels": relevant_eval_samples,
        "samples_without_relevant_labels": relevant_missing_samples,
        "relevant_label_coverage": float(relevant_eval_samples / total_samples) if total_samples > 0 else 0.0,
        "total_all_true_labels": total_all_true_labels,
        "total_relevant_true_labels": total_relevant_true_labels,
        "matched_top1_all_labels": matched_top1_all_total,
        f"matched_top{top_k}_all_labels": matched_topk_all_total,
        "matched_top1_relevant_labels": matched_top1_relevant_total,
        f"matched_top{top_k}_relevant_labels": matched_topk_relevant_total,
        "label_pickup_acc_top1_all_labels": label_pickup_acc_top1_all,
        f"label_pickup_acc_top{top_k}_all_labels": label_pickup_acc_topk_all,
        "label_pickup_acc_top1_relevant_labels": label_pickup_acc_top1_relevant,
        f"label_pickup_acc_top{top_k}_relevant_labels": label_pickup_acc_topk_relevant,
        "top1_acc_all_labels": all_top1_acc,
        f"top{top_k}_acc_all_labels": all_topk_acc,
        "top1_acc_relevant_labels": relevant_top1_acc,
        f"top{top_k}_acc_relevant_labels": relevant_topk_acc,
        "mean_all_true_labels_per_clip": float(pred_df["num_all_true_labels"].mean()) if not pred_df.empty else 0.0,
        "mean_relevant_true_labels_per_clip": (
            float(pred_df["num_relevant_true_labels"].mean()) if not pred_df.empty else 0.0
        ),
        "top1_micro_f1": micro_f1,
        "top1_macro_f1": macro_f1,
    }

    per_class_df = _build_per_class_df(pred_df, classes=classes, top_k=top_k)
    confusion_df = _build_confusion_df(pred_df, classes=classes)

    return {
        "summary": summary,
        "summary_df": pd.DataFrame([summary]),
        "predictions_df": pred_df,
        "per_class_df": per_class_df,
        "confusion_df": confusion_df,
        "classes": classes,
        "unique_all_labels": unique_all_labels,
        "unique_relevant_labels": unique_relevant_labels,
        "artifacts": artifacts,
    }


def evaluate_run(
    model_path: str | Path,
    *,
    audio_config_path: str | Path = "src/configs/audio_params.yaml",
    labels_config_path: str | Path = "src/configs/labels.yaml",
    test_root_override: Optional[str | Path] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    top_k: int = 3,
    max_samples: Optional[int] = None,
    device: str = "auto",
    show_progress: bool = True,
) -> dict[str, Any]:
    return _evaluate_run_impl(
        model_path,
        dataset_fallback="irmas",
        audio_config_path=audio_config_path,
        labels_config_path=labels_config_path,
        test_root_override=test_root_override,
        batch_size=batch_size,
        num_workers=num_workers,
        top_k=top_k,
        max_samples=max_samples,
        device=device,
        show_progress=show_progress,
    )


def evaluate_irmas_part1_run(
    model_path: str | Path,
    *,
    audio_config_path: str | Path = "src/configs/audio_params.yaml",
    labels_config_path: str | Path = "src/configs/labels.yaml",
    test_root_override: Optional[str | Path] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    top_k: int = 3,
    max_samples: Optional[int] = None,
    device: str = "auto",
    show_progress: bool = True,
) -> dict[str, Any]:
    return _evaluate_run_impl(
        model_path,
        dataset_fallback="irmas",
        audio_config_path=audio_config_path,
        labels_config_path=labels_config_path,
        test_root_override=test_root_override,
        batch_size=batch_size,
        num_workers=num_workers,
        top_k=top_k,
        max_samples=max_samples,
        device=device,
        show_progress=show_progress,
    )


def evaluate_film_run(
    model_path: str | Path,
    *,
    audio_config_path: str | Path = "src/configs/audio_params.yaml",
    labels_config_path: str | Path = "src/configs/labels.yaml",
    test_root_override: Optional[str | Path] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    top_k: int = 3,
    max_samples: Optional[int] = None,
    device: str = "auto",
    show_progress: bool = True,
) -> dict[str, Any]:
    return _evaluate_run_impl(
        model_path,
        dataset_fallback="chinese_instruments",
        audio_config_path=audio_config_path,
        labels_config_path=labels_config_path,
        test_root_override=test_root_override,
        batch_size=batch_size,
        num_workers=num_workers,
        top_k=top_k,
        max_samples=max_samples,
        device=device,
        show_progress=show_progress,
    )


def _compare_runs_impl(
    model_paths: Iterable[str | Path],
    *,
    dataset_fallback: str,
    test_root: Optional[str | Path],
    output_root: Optional[str | Path],
    batch_size: int,
    num_workers: int,
    top_k: int,
    max_samples: Optional[int],
    audio_config_path: str | Path,
    labels_config_path: str | Path,
    save_run_artifacts: bool,
    device: str,
    show_progress: bool,
    summary_filename: str,
    per_class_filename: str,
    predictions_suffix: str,
    confusion_suffix: str,
) -> dict[str, Any]:
    results_by_name: dict[str, Any] = {}
    summaries: list[dict[str, Any]] = []
    per_class_frames: list[pd.DataFrame] = []

    for model_path in model_paths:
        result = _evaluate_run_impl(
            model_path,
            dataset_fallback=dataset_fallback,
            audio_config_path=audio_config_path,
            labels_config_path=labels_config_path,
            test_root_override=test_root,
            batch_size=batch_size,
            num_workers=num_workers,
            top_k=top_k,
            max_samples=max_samples,
            device=device,
            show_progress=show_progress,
        )
        model_name = str(result["summary"]["model_name"])
        results_by_name[model_name] = result
        summaries.append(result["summary"])

        per_class = result["per_class_df"].copy()
        per_class.insert(0, "model_name", model_name)
        per_class_frames.append(per_class)

    comparison_df = pd.DataFrame(summaries).sort_values("model_name").reset_index(drop=True)
    per_class_comparison_df = (
        pd.concat(per_class_frames, ignore_index=True)
        if per_class_frames
        else pd.DataFrame(
            columns=[
                "model_name",
                "class_name",
                "n_occurrences",
                "top1_pickup_rate",
                f"top{max(1, int(top_k))}_pickup_rate",
            ]
        )
    )

    resolved_output = None
    if output_root:
        root = _repo_root()
        resolved_output = resolve_path(output_root, root)
        resolved_output.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(resolved_output / summary_filename, index=False)
        per_class_comparison_df.to_csv(resolved_output / per_class_filename, index=False)
        if save_run_artifacts:
            for model_name, result in results_by_name.items():
                safe_name = model_name.replace("/", "_")
                result["predictions_df"].to_csv(
                    resolved_output / f"{safe_name}{predictions_suffix}",
                    index=False,
                )
                result["confusion_df"].to_csv(
                    resolved_output / f"{safe_name}{confusion_suffix}",
                )

    return {
        "comparison_df": comparison_df,
        "per_class_comparison_df": per_class_comparison_df,
        "results_by_name": results_by_name,
        "output_root": str(resolved_output) if resolved_output else "",
    }


def compare_runs(
    model_paths: Iterable[str | Path],
    *,
    test_root: Optional[str | Path] = None,
    output_root: Optional[str | Path] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    top_k: int = 3,
    max_samples: Optional[int] = None,
    audio_config_path: str | Path = "src/configs/audio_params.yaml",
    labels_config_path: str | Path = "src/configs/labels.yaml",
    save_run_artifacts: bool = False,
    device: str = "auto",
    show_progress: bool = True,
    dataset_fallback: str = "irmas",
) -> dict[str, Any]:
    return _compare_runs_impl(
        model_paths,
        dataset_fallback=dataset_fallback,
        test_root=test_root,
        output_root=output_root,
        batch_size=batch_size,
        num_workers=num_workers,
        top_k=top_k,
        max_samples=max_samples,
        audio_config_path=audio_config_path,
        labels_config_path=labels_config_path,
        save_run_artifacts=save_run_artifacts,
        device=device,
        show_progress=show_progress,
        summary_filename="comparison_summary.csv",
        per_class_filename="comparison_per_class.csv",
        predictions_suffix="_predictions.csv",
        confusion_suffix="_confusion.csv",
    )


def compare_irmas_part1_runs(
    model_paths: Iterable[str | Path],
    *,
    test_root: Optional[str | Path] = None,
    cache_root: Optional[str | Path] = None,
    output_root: Optional[str | Path] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    top_k: int = 3,
    max_samples: Optional[int] = None,
    audio_config_path: str | Path = "src/configs/audio_params.yaml",
    labels_config_path: str | Path = "src/configs/labels.yaml",
    force_rebuild_features: bool = False,
    save_run_artifacts: bool = False,
    device: str = "auto",
    show_progress: bool = True,
) -> dict[str, Any]:
    _ = cache_root
    _ = force_rebuild_features

    return _compare_runs_impl(
        model_paths,
        dataset_fallback="irmas",
        test_root=test_root,
        output_root=output_root,
        batch_size=batch_size,
        num_workers=num_workers,
        top_k=top_k,
        max_samples=max_samples,
        audio_config_path=audio_config_path,
        labels_config_path=labels_config_path,
        save_run_artifacts=save_run_artifacts,
        device=device,
        show_progress=show_progress,
        summary_filename="comparison_summary.csv",
        per_class_filename="comparison_per_class.csv",
        predictions_suffix="_predictions.csv",
        confusion_suffix="_confusion.csv",
    )


def compare_film_runs(
    model_paths: Iterable[str | Path],
    *,
    test_root: Optional[str | Path] = None,
    output_root: Optional[str | Path] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    top_k: int = 3,
    max_samples: Optional[int] = None,
    audio_config_path: str | Path = "src/configs/audio_params.yaml",
    labels_config_path: str | Path = "src/configs/labels.yaml",
    save_run_artifacts: bool = False,
    device: str = "auto",
    show_progress: bool = True,
) -> dict[str, Any]:
    return _compare_runs_impl(
        model_paths,
        dataset_fallback="chinese_instruments",
        test_root=test_root,
        output_root=output_root,
        batch_size=batch_size,
        num_workers=num_workers,
        top_k=top_k,
        max_samples=max_samples,
        audio_config_path=audio_config_path,
        labels_config_path=labels_config_path,
        save_run_artifacts=save_run_artifacts,
        device=device,
        show_progress=show_progress,
        summary_filename="film_comparison_summary.csv",
        per_class_filename="film_comparison_per_class.csv",
        predictions_suffix="_film_predictions.csv",
        confusion_suffix="_film_confusion.csv",
    )
