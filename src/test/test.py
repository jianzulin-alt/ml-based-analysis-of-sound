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

from src.models.CNN import CNN
from src.models.CNN_DenseNet_121 import CNN_DenseNet_121
from src.preprocessing.features import (
    compute_stereo_cqt_db,
    compute_stereo_logmel_db,
    compute_stft_params,
)
from src.preprocessing.preprocessing import (
    conform_audio_duration,
    load_audio_as_stereo_and_resample,
    preprocess_loudness,
)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.preprocessing.features import (
    compute_stereo_cqt_db,
    compute_stereo_logmel_db,
    compute_stft_params,
)
from src.preprocessing.preprocessing import (
    conform_audio_duration,
    load_audio_as_stereo_and_resample,
    preprocess_loudness,
)



def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# TODO: add to utils
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
    keys = ["train_labels"]
    if dataset_name == "irmas":
        keys = ["irmas_labels", "train_labels"]
    for key in keys:
        labels = labels_cfg.get(key)
        if labels:
            classes = [str(x).strip().lower() for x in labels if x is not None]
            if classes:
                return classes
    raise ValueError(
        f"No labels found for dataset='{dataset_name}'. "
        f"Tried keys: {keys} in labels config."
    )


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
    raise ValueError(f"Unsupported backbone: {backbone}")


def _read_irmas_txt_labels(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    labels: list[str] = []
    for line in raw.splitlines():
        token = line.strip().lower().replace("\t", " ")
        token = token.split()[0] if token else ""
        token = token.strip()
        if token:
            labels.append(token)
    seen: set[str] = set()
    uniq: list[str] = []
    for lbl in labels:
        if lbl not in seen:
            seen.add(lbl)
            uniq.append(lbl)
    return uniq


@dataclass
class IRMASTestSample:
    wav_path: Path
    txt_path: Path
    labels: list[str]


def collect_irmas_test_samples(test_root: Path, valid_labels: set[str]) -> list[IRMASTestSample]:
    samples: list[IRMASTestSample] = []
    for wav_path in sorted(test_root.rglob("*.wav")):
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        labels = [lbl for lbl in _read_irmas_txt_labels(txt_path) if lbl in valid_labels]
        if not labels:
            continue
        samples.append(IRMASTestSample(wav_path=wav_path, txt_path=txt_path, labels=labels))
    return samples


class IRMASOnTheFlyDataset(Dataset):
    def __init__(
        self,
        samples: list[IRMASTestSample],
        classes: list[str],
        audio_cfg: dict,
        feature_mode: str,
    ) -> None:
        self.samples = samples
        self.classes = [c.strip().lower() for c in classes]
        self.label_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.audio_cfg = audio_cfg
        self.feature_mode = str(feature_mode).strip().lower()

        if self.feature_mode not in {"mel", "cqt", "mel_cqt"}:
            raise ValueError(f"Unsupported feature_mode for inference: {self.feature_mode}")

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
        self.loudness_norm = str(audio_cfg.get("loudness_norm", "none"))
        self.target_lufs = float(audio_cfg.get("target_lufs", -23.0))
        self.loudness_peak_limit = float(audio_cfg.get("loudness_peak_limit", 0.99))

    def __len__(self) -> int:
        return len(self.samples)

    def _extract_features(self, wav_path: Path) -> np.ndarray:
        stereo = load_audio_as_stereo_and_resample(wav_path, target_sr=self.sr)
        stereo = conform_audio_duration(stereo, self.sr, self.duration)
        stereo = preprocess_loudness(
            stereo,
            sr=self.sr,
            loudness_norm=self.loudness_norm,
            target_lufs=self.target_lufs,
            peak_limit=self.loudness_peak_limit,
        )

        if self.feature_mode == "mel":
            return compute_stereo_logmel_db(
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

        if self.feature_mode == "cqt":
            return compute_stereo_cqt_db(
                stereo,
                self.sr,
                n_bins=self.n_bins,
                bins_per_octave=self.bins_per_octave,
                hop_length=self.hop,
                fmin=self.fmin,
            )

        mel = compute_stereo_logmel_db(
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
        cqt = compute_stereo_cqt_db(
            stereo,
            self.sr,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
            hop_length=self.hop,
            fmin=self.fmin,
        )
        min_w = min(mel.shape[2], cqt.shape[2])
        return np.concatenate([mel[:, :, :min_w], cqt[:, :, :min_w]], axis=0).astype(np.float32)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        x_np = self._extract_features(sample.wav_path)
        x = torch.from_numpy(x_np).float()

        true_vec = torch.zeros(len(self.classes), dtype=torch.float32)
        for lbl in sample.labels:
            true_vec[self.label_to_idx[lbl]] = 1.0
        return x, int(idx), true_vec


def _resolve_model_artifacts(model_path: str | Path, root: Path) -> dict[str, Any]:
    resolved = resolve_path(model_path, root)
    if not resolved.exists():
        raise FileNotFoundError(f"Model path not found: {resolved}")

    if resolved.is_file():
        if resolved.suffix != ".pt":
            raise ValueError(f"Expected a .pt checkpoint file, got: {resolved}")
        run_dir = resolved.parent
        ckpt_path = resolved
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


def _load_run_config_with_fallback(
    run_config_path: Path,
    ckpt: dict[str, Any],
) -> dict[str, Any]:
    if run_config_path.exists():
        try:
            cfg = load_yaml(run_config_path)
            if isinstance(cfg, dict) and cfg:
                return cfg
        except Exception:
            pass
    cfg = ckpt.get("config", {})
    return cfg if isinstance(cfg, dict) else {}


def _resolve_test_root(
    dataset_name: str,
    datasets_cfg: dict,
    root: Path,
    test_root_override: Optional[str | Path] = None,
) -> Path:
    if test_root_override is not None and str(test_root_override).strip():
        return resolve_path(test_root_override, root)

    dataset_cfg = datasets_cfg.get(dataset_name, {})
    test_rel = dataset_cfg.get("test", "")
    if test_rel:
        return resolve_path(test_rel, root)

    if dataset_name == "irmas":
        return resolve_path("data/IRMAS/IRMAS-TestingData-Part1/Part1", root)

    raise ValueError(
        f"No test path configured for dataset='{dataset_name}'. "
        "Set datasets.<name>.test in audio config or pass test_root_override."
    )


def _f1(precision: float, recall: float) -> float:
    return float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


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
    root = _repo_root()
    artifacts = _resolve_model_artifacts(model_path, root)
    ckpt = torch.load(artifacts["checkpoint_path"], map_location="cpu", weights_only=False)
    run_cfg = _load_run_config_with_fallback(artifacts["run_config_path"], ckpt)
    resolved_cfg = run_cfg.get("resolved", {}) if isinstance(run_cfg.get("resolved"), dict) else {}

    audio_cfg_path = resolve_path(
        resolved_cfg.get("audio_config", str(audio_config_path)),
        root,
    )
    labels_cfg_path = resolve_path(
        resolved_cfg.get("labels_config", str(labels_config_path)),
        root,
    )

    audio_cfg_all = load_yaml(audio_cfg_path)
    labels_cfg = load_yaml(labels_cfg_path)

    dataset_name = str(run_cfg.get("dataset", "irmas")).strip().lower()
    feature_mode = str(run_cfg.get("feature_mode", "mel")).strip().lower()
    model_cfg = run_cfg.get("model", {}) if isinstance(run_cfg.get("model"), dict) else {}
    backbone = str(model_cfg.get("backbone", "cnn")).strip().lower()

    datasets_cfg = audio_cfg_all.get("datasets", {}) or {}
    if dataset_name not in datasets_cfg:
        raise ValueError(f"Dataset '{dataset_name}' not found in {audio_cfg_path}")
    audio_cfg = audio_cfg_all.get("audio", {}) or {}
    if not audio_cfg:
        raise ValueError(f"Missing 'audio' section in {audio_cfg_path}")

    test_root = _resolve_test_root(dataset_name, datasets_cfg, root, test_root_override=test_root_override)
    if not test_root.exists():
        raise FileNotFoundError(f"IRMAS test root not found: {test_root}")

    classes = choose_classes(labels_cfg, dataset_name)
    valid_labels = set(classes)
    label_to_idx = {label: i for i, label in enumerate(classes)}

    samples = collect_irmas_test_samples(test_root, valid_labels=valid_labels)
    if max_samples is not None:
        samples = samples[: max(0, int(max_samples))]
    if not samples:
        raise ValueError(f"No valid IRMAS test samples found under: {test_root}")

    infer_device = select_device(device)
    in_ch = 4 if feature_mode == "mel_cqt" else 2
    model = build_model(backbone=backbone, in_ch=in_ch, num_classes=len(classes), model_cfg=model_cfg)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=True)
    model = model.to(infer_device)
    model.eval()

    ds = IRMASOnTheFlyDataset(samples=samples, classes=classes, audio_cfg=audio_cfg, feature_mode=feature_mode)
    dl = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(infer_device.type == "cuda"),
    )

    rows: list[dict[str, Any]] = []
    top_k = max(1, int(top_k))

    tp_total = 0
    fp_total = 0
    fn_total = 0
    tp_by_class = np.zeros(len(classes), dtype=np.int64)
    fp_by_class = np.zeros(len(classes), dtype=np.int64)
    fn_by_class = np.zeros(len(classes), dtype=np.int64)

    iterator: Iterable = dl
    if show_progress:
        iterator = tqdm(dl, total=len(dl), desc="IRMAS test eval", leave=False)

    with torch.no_grad():
        for x, sample_indices, true_vecs in iterator:
            x = x.to(infer_device, non_blocking=(infer_device.type == "cuda"))
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu()

            pred_top1 = torch.argmax(probs, dim=1)
            pred_topk = torch.topk(probs, k=min(top_k, probs.size(1)), dim=1).indices

            sample_idx_np = sample_indices.cpu().numpy().tolist()
            pred1_np = pred_top1.cpu().numpy().tolist()
            true_vec_np = true_vecs.cpu().numpy()
            pred_topk_np = pred_topk.cpu().numpy()
            probs_np = probs.numpy()

            for i, sample_idx in enumerate(sample_idx_np):
                sample = samples[sample_idx]
                true_labels = sample.labels
                true_label_set = set(true_labels)
                pred_idx = int(pred1_np[i])
                pred_label = classes[pred_idx]
                topk_labels = [classes[int(k)] for k in pred_topk_np[i].tolist()]
                matched_topk_labels = sorted(true_label_set.intersection(topk_labels))

                top1_hit_any = pred_label in true_label_set
                topk_hit_any = any(lbl in true_label_set for lbl in topk_labels)

                true_indices = np.flatnonzero(true_vec_np[i] > 0.0).tolist()
                if pred_idx in true_indices:
                    tp_total += 1
                    tp_by_class[pred_idx] += 1
                else:
                    fp_total += 1
                    fp_by_class[pred_idx] += 1
                fn_count = len(true_indices) - (1 if pred_idx in true_indices else 0)
                fn_total += fn_count
                for true_idx in true_indices:
                    if true_idx != pred_idx:
                        fn_by_class[true_idx] += 1

                rows.append(
                    {
                        "wav_path": str(sample.wav_path),
                        "txt_path": str(sample.txt_path),
                        "all_true_labels": "|".join(true_labels),
                        "pred_top1_label": pred_label,
                        "pred_top1_conf": float(probs_np[i, pred_idx]),
                        "pred_topk_labels": "|".join(topk_labels),
                        "top1_hit_any": bool(top1_hit_any),
                        f"top{top_k}_hit_any": bool(topk_hit_any),
                        "matched_topk_labels": "|".join(matched_topk_labels),
                        "matched_top1_label_count": int(top1_hit_any),
                        f"matched_top{top_k}_label_count": int(len(matched_topk_labels)),
                        "num_true_labels": int(np.sum(true_vec_np[i] > 0)),
                    }
                )

    pred_df = pd.DataFrame(rows)

    micro_precision = float(tp_total / (tp_total + fp_total)) if (tp_total + fp_total) > 0 else 0.0
    micro_recall = float(tp_total / (tp_total + fn_total)) if (tp_total + fn_total) > 0 else 0.0
    micro_f1 = _f1(micro_precision, micro_recall)

    class_f1_vals: list[float] = []
    for class_idx in range(len(classes)):
        p_denom = int(tp_by_class[class_idx] + fp_by_class[class_idx])
        r_denom = int(tp_by_class[class_idx] + fn_by_class[class_idx])
        precision = float(tp_by_class[class_idx] / p_denom) if p_denom > 0 else 0.0
        recall = float(tp_by_class[class_idx] / r_denom) if r_denom > 0 else 0.0
        class_f1_vals.append(_f1(precision, recall))
    macro_f1 = float(np.mean(class_f1_vals)) if class_f1_vals else 0.0

    any_top1_acc = float(pred_df["top1_hit_any"].mean()) if not pred_df.empty else 0.0
    any_topk_col = f"top{top_k}_hit_any"
    any_topk_acc = float(pred_df[any_topk_col].mean()) if not pred_df.empty else 0.0
    total_true_labels = int(pred_df["num_true_labels"].sum()) if not pred_df.empty else 0
    matched_top1_total = int(pred_df["matched_top1_label_count"].sum()) if not pred_df.empty else 0
    matched_topk_total = int(pred_df[f"matched_top{top_k}_label_count"].sum()) if not pred_df.empty else 0
    label_pickup_top1 = float(matched_top1_total / total_true_labels) if total_true_labels > 0 else 0.0
    label_pickup_topk = float(matched_topk_total / total_true_labels) if total_true_labels > 0 else 0.0

    summary = {
        "model_name": artifacts["model_name"],
        "checkpoint_path": str(artifacts["checkpoint_path"]),
        "run_config_path": str(artifacts["run_config_path"]),
        "dataset": dataset_name,
        "feature_mode": feature_mode,
        "backbone": backbone,
        "test_root": str(test_root),
        "device": str(infer_device),
        "samples": int(len(pred_df)),
        "classes": int(len(classes)),
        "top1_macro_f1": macro_f1,
        "top1_micro_f1": micro_f1,
        "any_label_top1_acc": any_top1_acc,
        f"any_label_top{top_k}_acc": any_topk_acc,
        "total_true_labels": total_true_labels,
        "matched_top1_labels": matched_top1_total,
        f"matched_top{top_k}_labels": matched_topk_total,
        "label_pickup_acc_top1": label_pickup_top1,
        f"label_pickup_acc_top{top_k}": label_pickup_topk,
        "mean_true_labels_per_clip": float(pred_df["num_true_labels"].mean()) if not pred_df.empty else 0.0,
    }

    if pred_df.empty:
        per_class_df = pd.DataFrame(
            columns=[
                "class_name",
                "n_occurrences",
                "top1_pickup_rate",
                f"top{top_k}_pickup_rate",
            ]
        )
        confusion_df = pd.DataFrame(0, index=classes, columns=classes)
    else:
        per_class_rows: list[dict[str, Any]] = []
        for class_name in classes:
            class_idx = label_to_idx[class_name]
            mask = pred_df["all_true_labels"].apply(
                lambda s, idx=class_idx: (classes[idx] in set(str(s).split("|"))) if str(s) else False
            )
            n_occurrences = int(mask.sum())
            if n_occurrences == 0:
                continue

            class_rows = pred_df.loc[mask]
            top1_pickup = float((class_rows["pred_top1_label"] == class_name).mean())
            topk_pickup = float(
                class_rows["pred_topk_labels"].apply(
                    lambda s, label=class_name: (label in set(str(s).split("|"))) if str(s) else False
                ).mean()
            )
            per_class_rows.append(
                {
                    "class_name": class_name,
                    "n_occurrences": n_occurrences,
                    "top1_pickup_rate": top1_pickup,
                    f"top{top_k}_pickup_rate": topk_pickup,
                }
            )
        per_class_df = pd.DataFrame(per_class_rows).sort_values("class_name").reset_index(drop=True)

        confusion_df = pd.DataFrame(0, index=classes, columns=classes)
        for _, row in pred_df.iterrows():
            pred_lbl = str(row["pred_top1_label"])
            true_labels = [lbl for lbl in str(row["all_true_labels"]).split("|") if lbl]
            for true_lbl in true_labels:
                if true_lbl in confusion_df.index and pred_lbl in confusion_df.columns:
                    confusion_df.loc[true_lbl, pred_lbl] += 1

    return {
        "summary": summary,
        "summary_df": pd.DataFrame([summary]),
        "predictions_df": pred_df,
        "per_class_df": per_class_df,
        "confusion_df": confusion_df,
        "classes": classes,
        "artifacts": artifacts,
    }


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
) -> dict[str, Any]:
    _ = cache_root
    _ = force_rebuild_features

    results_by_name: dict[str, Any] = {}
    summaries: list[dict[str, Any]] = []
    per_class_frames: list[pd.DataFrame] = []

    for model_path in model_paths:
        result = evaluate_irmas_part1_run(
            model_path,
            audio_config_path=audio_config_path,
            labels_config_path=labels_config_path,
            test_root_override=test_root,
            batch_size=batch_size,
            num_workers=num_workers,
            top_k=top_k,
            max_samples=max_samples,
            device=device,
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
            ]
        )
    )

    resolved_output = None
    if output_root:
        root = _repo_root()
        resolved_output = resolve_path(output_root, root)
        resolved_output.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(resolved_output / "comparison_summary.csv", index=False)
        per_class_comparison_df.to_csv(resolved_output / "comparison_per_class.csv", index=False)
        if save_run_artifacts:
            for model_name, result in results_by_name.items():
                safe_name = model_name.replace("/", "_")
                result["predictions_df"].to_csv(
                    resolved_output / f"{safe_name}_predictions.csv", index=False
                )
                result["confusion_df"].to_csv(
                    resolved_output / f"{safe_name}_confusion.csv"
                )

    return {
        "comparison_df": comparison_df,
        "per_class_comparison_df": per_class_comparison_df,
        "results_by_name": results_by_name,
        "output_root": str(resolved_output) if resolved_output else "",
    }



@dataclass
class FilmTestSample:
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
        token = token.strip()
        if token:
            labels.append(token)
    seen: set[str] = set()
    uniq: list[str] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            uniq.append(label)
    return uniq


def collect_film_test_samples(
    test_root: Path,
    relevant_labels_set: set[str],
) -> list[FilmTestSample]:
    samples: list[FilmTestSample] = []
    for wav_path in sorted(test_root.rglob("*.wav")):
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        all_labels = _read_txt_labels(txt_path)
        if not all_labels:
            continue
        relevant_labels = [lbl for lbl in all_labels if lbl in relevant_labels_set]
        samples.append(
            FilmTestSample(
                wav_path=wav_path,
                txt_path=txt_path,
                all_labels=all_labels,
                relevant_labels=relevant_labels,
            )
        )
    return samples


class FilmOnTheFlyDataset(Dataset):
    def __init__(
        self,
        samples: list[FilmTestSample],
        audio_cfg: dict,
        feature_mode: str,
    ) -> None:
        self.samples = samples
        self.audio_cfg = audio_cfg
        self.feature_mode = str(feature_mode).strip().lower()
        if self.feature_mode not in {"mel", "cqt", "mel_cqt"}:
            raise ValueError(f"Unsupported feature_mode for inference: {self.feature_mode}")

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
        self.loudness_norm = str(audio_cfg.get("loudness_norm", "none"))
        self.target_lufs = float(audio_cfg.get("target_lufs", -23.0))
        self.loudness_peak_limit = float(audio_cfg.get("loudness_peak_limit", 0.99))

    def __len__(self) -> int:
        return len(self.samples)

    def _extract_features(self, wav_path: Path) -> np.ndarray:
        stereo = load_audio_as_stereo_and_resample(wav_path, target_sr=self.sr)
        stereo = conform_audio_duration(stereo, self.sr, self.duration)
        stereo = preprocess_loudness(
            stereo,
            sr=self.sr,
            loudness_norm=self.loudness_norm,
            target_lufs=self.target_lufs,
            peak_limit=self.loudness_peak_limit,
        )

        if self.feature_mode == "mel":
            return compute_stereo_logmel_db(
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

        if self.feature_mode == "cqt":
            return compute_stereo_cqt_db(
                stereo,
                self.sr,
                n_bins=self.n_bins,
                bins_per_octave=self.bins_per_octave,
                hop_length=self.hop,
                fmin=self.fmin,
            )

        mel = compute_stereo_logmel_db(
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
        cqt = compute_stereo_cqt_db(
            stereo,
            self.sr,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
            hop_length=self.hop,
            fmin=self.fmin,
        )
        min_w = min(mel.shape[2], cqt.shape[2])
        return np.concatenate([mel[:, :, :min_w], cqt[:, :, :min_w]], axis=0).astype(np.float32)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        x_np = self._extract_features(sample.wav_path)
        x = torch.from_numpy(x_np).float()
        return x, int(idx)


def _resolve_test_root(
    dataset_name: str,
    datasets_cfg: dict,
    root: Path,
    test_root_override: Optional[str | Path] = None,
) -> Path:
    if test_root_override is not None and str(test_root_override).strip():
        return resolve_path(test_root_override, root)

    dataset_cfg = datasets_cfg.get(dataset_name, {})
    test_rel = dataset_cfg.get("test", "")
    if test_rel:
        return resolve_path(test_rel, root)

    if dataset_name == "chinese_instruments":
        return resolve_path("data/test/a-touch-of-zen/", root)

    raise ValueError(
        f"No test path configured for dataset='{dataset_name}'. "
        "Set datasets.<name>.test in audio config or pass test_root_override."
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
    """
    Evaluate a trained Chinese-instrument model on the film test set with on-the-fly DSP.

    Notes on `top_k`:
    - We compute class probabilities per clip via `softmax(logits)`.
    - `top_k` means: take the K classes with highest probabilities for each clip.
    - We then report additional "hit" metrics based on whether any of those top-K
      predicted labels intersects with the clip's ground-truth label list.
    - We also compute label-pickup accuracy as:
      (number of picked-up labels) / (total ground-truth labels).
    - `top_k` does NOT change the top-1 prediction itself. Top-1 still uses argmax.
    """
    root = _repo_root()
    artifacts = _resolve_model_artifacts(model_path, root)
    ckpt = torch.load(artifacts["checkpoint_path"], map_location="cpu", weights_only=False)
    run_cfg = _load_run_config_with_fallback(artifacts["run_config_path"], ckpt)
    resolved_cfg = run_cfg.get("resolved", {}) if isinstance(run_cfg.get("resolved"), dict) else {}

    audio_cfg_path = resolve_path(
        resolved_cfg.get("audio_config", str(audio_config_path)),
        root,
    )
    labels_cfg_path = resolve_path(
        resolved_cfg.get("labels_config", str(labels_config_path)),
        root,
    )

    audio_cfg_all = load_yaml(audio_cfg_path)
    labels_cfg = load_yaml(labels_cfg_path)

    dataset_name = str(run_cfg.get("dataset", "chinese_instruments")).strip().lower()
    feature_mode = str(run_cfg.get("feature_mode", "mel")).strip().lower()
    model_cfg = run_cfg.get("model", {}) if isinstance(run_cfg.get("model"), dict) else {}
    backbone = str(model_cfg.get("backbone", "cnn")).strip().lower()

    datasets_cfg = audio_cfg_all.get("datasets", {}) or {}
    if dataset_name not in datasets_cfg:
        raise ValueError(f"Dataset '{dataset_name}' not found in {audio_cfg_path}")
    audio_cfg = audio_cfg_all.get("audio", {}) or {}
    if not audio_cfg:
        raise ValueError(f"Missing 'audio' section in {audio_cfg_path}")

    test_root = _resolve_test_root(dataset_name, datasets_cfg, root, test_root_override=test_root_override)
    if not test_root.exists():
        raise FileNotFoundError(f"Film test root not found: {test_root}")

    classes = choose_classes(labels_cfg, dataset_name)
    relevant_labels_set = set(classes)

    samples = collect_film_test_samples(test_root, relevant_labels_set=relevant_labels_set)
    if max_samples is not None:
        samples = samples[: max(0, int(max_samples))]
    if not samples:
        raise ValueError(f"No valid film test samples found under: {test_root}")

    infer_device = select_device(device)
    in_ch = 4 if feature_mode == "mel_cqt" else 2
    model = build_model(backbone=backbone, in_ch=in_ch, num_classes=len(classes), model_cfg=model_cfg)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=True)
    model = model.to(infer_device)
    model.eval()

    ds = FilmOnTheFlyDataset(samples=samples, audio_cfg=audio_cfg, feature_mode=feature_mode)
    dl = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(infer_device.type == "cuda"),
    )

    rows: list[dict[str, Any]] = []
    # Safety clamp: top_k must be >= 1.
    top_k = max(1, int(top_k))

    iterator: Iterable = dl
    if show_progress:
        iterator = tqdm(dl, total=len(dl), desc="Film test eval", leave=False)

    with torch.no_grad():
        for x, sample_indices in iterator:
            x = x.to(infer_device, non_blocking=(infer_device.type == "cuda"))
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu()
            pred_top1 = torch.argmax(probs, dim=1)
            # Top-K indices are the K highest-probability classes per sample.
            # If K > num_classes, use num_classes to avoid overflow.
            pred_topk = torch.topk(probs, k=min(top_k, probs.size(1)), dim=1).indices

            sample_idx_np = sample_indices.cpu().numpy().tolist()
            pred1_np = pred_top1.cpu().numpy().tolist()
            pred_topk_np = pred_topk.cpu().numpy()
            probs_np = probs.numpy()

            for i, sample_idx in enumerate(sample_idx_np):
                sample = samples[sample_idx]
                pred_idx = int(pred1_np[i])
                pred_label = classes[pred_idx]
                topk_labels = [classes[int(k)] for k in pred_topk_np[i].tolist()]

                all_label_set = set(sample.all_labels)
                relevant_label_set = set(sample.relevant_labels)
                has_relevant = bool(sample.relevant_labels)

                top1_hit_all = pred_label in all_label_set
                topk_hit_all = any(lbl in all_label_set for lbl in topk_labels)

                top1_hit_relevant = (pred_label in relevant_label_set) if has_relevant else np.nan
                topk_hit_relevant = (
                    any(lbl in relevant_label_set for lbl in topk_labels) if has_relevant else np.nan
                )

                matched_topk_all = sorted(all_label_set.intersection(topk_labels))
                matched_topk_relevant = sorted(relevant_label_set.intersection(topk_labels))

                rows.append(
                    {
                        "wav_file": sample.wav_path.name,
                        "txt_file": sample.txt_path.name,
                        "all_true_labels": "|".join(sample.all_labels),
                        "relevant_true_labels": "|".join(sample.relevant_labels),
                        "pred_top1_label": pred_label,
                        "pred_top1_conf": float(probs_np[i, pred_idx]),
                        "pred_topk_labels": "|".join(topk_labels),
                        "num_all_true_labels": int(len(sample.all_labels)),
                        "num_relevant_true_labels": int(len(sample.relevant_labels)),
                        "has_relevant_labels": bool(has_relevant),
                        "top1_hit_all_labels": bool(top1_hit_all),
                        f"top{top_k}_hit_all_labels": bool(topk_hit_all),
                        "top1_hit_relevant_labels": top1_hit_relevant,
                        f"top{top_k}_hit_relevant_labels": topk_hit_relevant,
                        "matched_topk_all_labels": "|".join(matched_topk_all),
                        "matched_topk_relevant_labels": "|".join(matched_topk_relevant),
                        "matched_top1_all_label_count": int(top1_hit_all),
                        f"matched_top{top_k}_all_label_count": int(len(matched_topk_all)),
                        "matched_top1_relevant_label_count": int(pred_label in relevant_label_set),
                        f"matched_top{top_k}_relevant_label_count": int(len(matched_topk_relevant)),
                    }
                )

    pred_df = pd.DataFrame(rows)
    total_samples = int(len(pred_df))
    relevant_mask = pred_df["has_relevant_labels"] if not pred_df.empty else pd.Series([], dtype=bool)
    relevant_eval_samples = int(relevant_mask.sum()) if not pred_df.empty else 0
    relevant_missing_samples = total_samples - relevant_eval_samples

    all_top1_acc = float(pred_df["top1_hit_all_labels"].mean()) if not pred_df.empty else 0.0
    all_topk_col = f"top{top_k}_hit_all_labels"
    all_topk_acc = float(pred_df[all_topk_col].mean()) if not pred_df.empty else 0.0

    rel_top1_acc = (
        float(pd.to_numeric(pred_df.loc[relevant_mask, "top1_hit_relevant_labels"], errors="coerce").mean())
        if relevant_eval_samples > 0
        else float("nan")
    )
    rel_topk_col = f"top{top_k}_hit_relevant_labels"
    rel_topk_acc = (
        float(pd.to_numeric(pred_df.loc[relevant_mask, rel_topk_col], errors="coerce").mean())
        if relevant_eval_samples > 0
        else float("nan")
    )

    total_all_true_labels = int(pred_df["num_all_true_labels"].sum()) if not pred_df.empty else 0
    total_relevant_true_labels = int(pred_df["num_relevant_true_labels"].sum()) if not pred_df.empty else 0
    matched_top1_all_total = (
        int(pred_df["matched_top1_all_label_count"].sum()) if not pred_df.empty else 0
    )
    matched_topk_all_total = (
        int(pred_df[f"matched_top{top_k}_all_label_count"].sum()) if not pred_df.empty else 0
    )
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
            lbl
            for value in pred_df.get("all_true_labels", pd.Series([], dtype=str)).tolist()
            for lbl in str(value).split("|")
            if lbl
        }
    )
    unique_relevant_labels = sorted(
        {
            lbl
            for value in pred_df.get("relevant_true_labels", pd.Series([], dtype=str)).tolist()
            for lbl in str(value).split("|")
            if lbl
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
        "relevant_label_coverage": (
            float(relevant_eval_samples / total_samples) if total_samples > 0 else 0.0
        ),
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
        "top1_acc_relevant_labels": rel_top1_acc,
        f"top{top_k}_acc_relevant_labels": rel_topk_acc,
        "mean_all_true_labels_per_clip": (
            float(pred_df["num_all_true_labels"].mean()) if not pred_df.empty else 0.0
        ),
        "mean_relevant_true_labels_per_clip": (
            float(pred_df["num_relevant_true_labels"].mean()) if not pred_df.empty else 0.0
        ),
    }

    if relevant_eval_samples == 0:
        per_class_df = pd.DataFrame(
            columns=[
                "class_name",
                "n_occurrences",
                "top1_pickup_rate",
                f"top{top_k}_pickup_rate",
            ]
        )
        confusion_df = pd.DataFrame(0, index=classes, columns=classes)
    else:
        relevant_df = pred_df.loc[relevant_mask].copy()
        rows_per_class: list[dict[str, Any]] = []
        for class_name in classes:
            mask = relevant_df["relevant_true_labels"].apply(
                lambda s: class_name in set(str(s).split("|")) if str(s) else False
            )
            class_occ = int(mask.sum())
            if class_occ == 0:
                continue
            class_rows = relevant_df.loc[mask]
            top1_pickup = float((class_rows["pred_top1_label"] == class_name).mean())
            topk_pickup = float(
                class_rows["pred_topk_labels"].apply(
                    lambda s: class_name in set(str(s).split("|")) if str(s) else False
                ).mean()
            )
            rows_per_class.append(
                {
                    "class_name": class_name,
                    "n_occurrences": class_occ,
                    "top1_pickup_rate": top1_pickup,
                    f"top{top_k}_pickup_rate": topk_pickup,
                }
            )
        per_class_df = pd.DataFrame(rows_per_class).sort_values("class_name").reset_index(drop=True)

        confusion_df = pd.DataFrame(0, index=classes, columns=classes)
        for _, row in relevant_df.iterrows():
            pred_lbl = str(row["pred_top1_label"])
            true_labels = [x for x in str(row["relevant_true_labels"]).split("|") if x]
            for true_lbl in true_labels:
                if true_lbl in confusion_df.index and pred_lbl in confusion_df.columns:
                    confusion_df.loc[true_lbl, pred_lbl] += 1

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
) -> dict[str, Any]:
    results_by_name: dict[str, Any] = {}
    summaries: list[dict[str, Any]] = []
    per_class_frames: list[pd.DataFrame] = []

    for model_path in model_paths:
        result = evaluate_film_run(
            model_path,
            audio_config_path=audio_config_path,
            labels_config_path=labels_config_path,
            test_root_override=test_root,
            batch_size=batch_size,
            num_workers=num_workers,
            top_k=top_k,
            max_samples=max_samples,
            device=device,
        )
        model_name = str(result["summary"]["model_name"])
        results_by_name[model_name] = result
        summaries.append(result["summary"])

        per_class = result["per_class_df"].copy()
        per_class.insert(0, "model_name", model_name)
        per_class_frames.append(per_class)

    comparison_df = pd.DataFrame(summaries).sort_values("model_name").reset_index(drop=True)


    resolved_output = None
    if output_root:
        root = _repo_root()
        resolved_output = resolve_path(output_root, root)
        resolved_output.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(resolved_output / "film_comparison_summary.csv", index=False)
        # per_class_comparison_df.to_csv(resolved_output / "film_comparison_per_class.csv", index=False)
        if save_run_artifacts:
            for model_name, result in results_by_name.items():
                safe_name = model_name.replace("/", "_")
                result["predictions_df"].to_csv(
                    resolved_output / f"{safe_name}_film_predictions.csv", index=False
                )
                result["confusion_df"].to_csv(
                    resolved_output / f"{safe_name}_film_confusion.csv"
                )

    return {
        "comparison_df": comparison_df,
        # "per_class_comparison_df": per_class_comparison_df,
        "results_by_name": results_by_name,
        "output_root": str(resolved_output) if resolved_output else "",
    }
