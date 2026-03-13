from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset, random_split

from src.data_loader import UniversalAudioDataset
from src.models.CNN import CNN
from src.models.CNN_DenseNet_121 import CNN_DenseNet_121
from src.train.trainer import AudioTrainer, collate_fn_padd, get_device, load_checkpoint, seed_everything

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


def normalize_feature_mode(feature_mode: str) -> str:
    raw = str(feature_mode).strip().lower()
    if raw not in {"mel", "cqt"}:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")
    return raw


def choose_classes(labels_cfg: dict, dataset_name: str) -> List[str]:
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


def _candidate_manifest_paths(dataset_name: str, dataset_cfg: dict, suffix: str, root: Path) -> List[Path]:
    manifest_base = resolve_path(dataset_cfg["manifest"], root)
    parent = manifest_base.parent
    stem = manifest_base.stem
    return [
        parent / f"{dataset_name}_train_{suffix}.csv",
        parent / f"{stem}_{suffix}.csv",
    ]


def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _prepare_manifest_for_dataset(
    manifest_path: Path,
    run_dir: Path,
    *,
    ensure_filepath_from_cqt: bool,
    ensure_merge_keys: bool,
) -> Path:
    """
    Normalise manifest columns so UniversalAudioDataset can consume diverse CSV formats.
    """
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


def resolve_feature_manifests(
    feature_mode: str,
    dataset_name: str,
    dataset_cfg: dict,
    task_mode: str,
    root: Path,
    run_dir: Path,
) -> Tuple[Path, Optional[Path]]:
    """
    Returns (primary_manifest, cqt_manifest_if_needed).
    """
    # Multi-label manifest preference is temporarily disabled.
    # prefer_mixed = task_mode == "multi_label"
    _ = task_mode

    if feature_mode == "mel":
        candidates = []
        # if prefer_mixed:
        #     candidates.extend(_candidate_manifest_paths(dataset_name, dataset_cfg, "mels_mixed", root))
        candidates.extend(_candidate_manifest_paths(dataset_name, dataset_cfg, "mels", root))
        mel_path = _first_existing(candidates)
        if mel_path is None:
            raise FileNotFoundError(f"Could not find mel manifest. Tried: {[str(p) for p in candidates]}")
        mel_path = _prepare_manifest_for_dataset(
            mel_path, run_dir, ensure_filepath_from_cqt=False, ensure_merge_keys=False
        )
        return mel_path, None

    if feature_mode == "cqt":
        candidates = []
        # if prefer_mixed:
        #     candidates.extend(_candidate_manifest_paths(dataset_name, dataset_cfg, "cqt_mixed", root))
        candidates.extend(_candidate_manifest_paths(dataset_name, dataset_cfg, "cqt", root))
        cqt_path = _first_existing(candidates)
        if cqt_path is None:
            raise FileNotFoundError(f"Could not find cqt manifest. Tried: {[str(p) for p in candidates]}")
        cqt_path = _prepare_manifest_for_dataset(
            cqt_path, run_dir, ensure_filepath_from_cqt=True, ensure_merge_keys=False
        )
        return cqt_path, None

    # mel_cqt
    mel_candidates = []
    cqt_candidates = []
    # if prefer_mixed:
    #     mel_candidates.extend(_candidate_manifest_paths(dataset_name, dataset_cfg, "mels_mixed", root))
    #     cqt_candidates.extend(_candidate_manifest_paths(dataset_name, dataset_cfg, "cqt_mixed", root))
    mel_candidates.extend(_candidate_manifest_paths(dataset_name, dataset_cfg, "mels", root))
    cqt_candidates.extend(_candidate_manifest_paths(dataset_name, dataset_cfg, "cqt", root))

    mel_path = _first_existing(mel_candidates)
    cqt_path = _first_existing(cqt_candidates)
    if mel_path is None or cqt_path is None:
        raise FileNotFoundError(
            "Could not find mel/cqt manifests for mel_cqt mode.\n"
            f"mel tried: {[str(p) for p in mel_candidates]}\n"
            f"cqt tried: {[str(p) for p in cqt_candidates]}"
        )

    mel_path = _prepare_manifest_for_dataset(
        mel_path, run_dir, ensure_filepath_from_cqt=False, ensure_merge_keys=True
    )
    cqt_path = _prepare_manifest_for_dataset(
        cqt_path, run_dir, ensure_filepath_from_cqt=True, ensure_merge_keys=True
    )
    return mel_path, cqt_path


def load_saved_split_indices(split_indices_path: Path, dataset_size: int) -> Tuple[List[int], List[int]]:
    saved = torch.load(split_indices_path, map_location="cpu", weights_only=False)
    if "train_indices" not in saved or "val_indices" not in saved:
        raise KeyError(
            f"Split file must contain 'train_indices' and 'val_indices': {split_indices_path}"
        )

    train_indices = [int(i) for i in saved["train_indices"]]
    val_indices = [int(i) for i in saved["val_indices"]]
    all_indices = train_indices + val_indices

    if not train_indices or not val_indices:
        raise ValueError(f"Saved split must contain non-empty train and val subsets: {split_indices_path}")
    if any(i < 0 or i >= dataset_size for i in all_indices):
        raise ValueError(
            f"Saved split indices in {split_indices_path} do not match the current dataset size ({dataset_size})."
        )
    if len(all_indices) != dataset_size or len(set(all_indices)) != dataset_size:
        raise ValueError(
            f"Saved split indices in {split_indices_path} do not cover the current dataset exactly."
        )

    return train_indices, val_indices


def infer_epochs_no_improve(history: dict | None, best_val_loss: float) -> int:
    if not isinstance(history, dict):
        return 0

    val_history = history.get("val_loss", [])
    if not isinstance(val_history, list) or not val_history:
        return 0

    trailing = 0
    for loss in reversed(val_history):
        if float(loss) <= best_val_loss + 1e-8:
            break
        trailing += 1
    return trailing


def build_model(backbone: str, in_ch: int, num_classes: int, model_cfg: dict) -> nn.Module:
    name = str(backbone).strip().lower()
    dropout = float(model_cfg.get("dropout", 0.3))

    if name == "cnn":
        return CNN(in_ch=in_ch, num_classes=num_classes, p_drop=dropout)
    if name in {"cnn_densenet_121"}:
        return CNN_DenseNet_121(in_ch=in_ch, num_classes=num_classes, p_drop=dropout)
    raise ValueError(f"Unsupported backbone: {backbone}")

def load_pretrained_for_finetuning(
    model: nn.Module, pretrained_path: Path, current_num_classes: int, device: str
) -> nn.Module:
    """
    Load pretrained weights while skipping mismatched classifier heads.
    """
    print(f"Loading pretrained weights from {pretrained_path}...")
    ckpt = torch.load(pretrained_path, map_location=device)
    pretrained_dict = ckpt.get("model_state", ckpt)
    model_dict = model.state_dict()

    matched = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(matched)
    model.load_state_dict(model_dict)
    print(f"Restored {len(matched)} layers. New head classes: {current_num_classes}")
    return model


def write_history_csv(history: dict, out_path: Path) -> None:
    keys = sorted(history.keys())
    n_rows = max((len(v) for v in history.values() if isinstance(v, list)), default=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(n_rows):
            row = []
            for k in keys:
                vals = history.get(k, [])
                row.append(vals[i] if i < len(vals) else "")
            writer.writerow(row)

def count_model_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YAML-driven training entrypoint for single-label audio models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="src/configs/train_params.yaml", help="Training run config YAML.")
    parser.add_argument("--audio_config", default="src/configs/audio_params.yaml", help="Audio/data config YAML.")
    parser.add_argument("--labels_config", default="src/configs/labels.yaml", help="Labels config YAML.")
    parser.add_argument(
        "--output_dir",
        default="",
        help="Optional checkpoint directory override. Default: src/models/saved_weights/<experiment_name>",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume full training state from <output_dir>/last.pt or the default run directory.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Setup everything and exit before training.")
    args = parser.parse_args()

    root = _repo_root()
    train_cfg_path = resolve_path(args.config, root)
    audio_cfg_path = resolve_path(args.audio_config, root)
    labels_cfg_path = resolve_path(args.labels_config, root)

    config = load_yaml(train_cfg_path)
    audio_cfg = load_yaml(audio_cfg_path)
    labels_cfg = load_yaml(labels_cfg_path)

    experiment_name = str(config.get("experiment_name", "exp")).strip()
    task_mode = str(config.get("task_mode", "single_label")).strip().lower()
    # if task_mode not in {"single_label", "multi_label"}:
    #     raise ValueError(f"Unsupported task_mode: {task_mode}")
    if task_mode != "single_label":
        raise ValueError("Only task_mode='single_label' is currently supported.")
    feature_mode = normalize_feature_mode(config.get("feature_mode", "mel"))
    dataset_name = str(config.get("dataset", "irmas")).strip()
    model_cfg = config.get("model", {}) or {}
    tr_cfg = config.get("training", {}) or {}

    datasets_cfg = audio_cfg.get("datasets", {}) or {}
    if dataset_name not in datasets_cfg:
        raise ValueError(f"Dataset '{dataset_name}' not found in {audio_cfg_path}")
    dataset_cfg = datasets_cfg[dataset_name]

    classes = choose_classes(labels_cfg, dataset_name)
    n_classes = len(classes)

    print("=" * 88)
    print("Training run setup")
    print(f"Experiment: {experiment_name}")
    print(f"Task mode: {task_mode} | Feature mode: {feature_mode} | Dataset: {dataset_name}")
    print(f"Configs: train={train_cfg_path} | audio={audio_cfg_path} | labels={labels_cfg_path}")
    print(f"Classes ({n_classes}): {', '.join(classes)}")

    default_out = root / "src" / "models" / "saved_weights" / experiment_name
    run_dir = resolve_path(args.output_dir, root) if args.output_dir else default_out
    run_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = run_dir / "last.pt"
    if args.resume and not resume_checkpoint.exists():
        raise FileNotFoundError(
            f"Resume requested but checkpoint not found: {resume_checkpoint}"
        )

    primary_manifest, cqt_manifest = resolve_feature_manifests(
        feature_mode=feature_mode,
        dataset_name=dataset_name,
        dataset_cfg=dataset_cfg,
        task_mode=task_mode,
        root=root,
        run_dir=run_dir,
    )
    print(f"Using primary manifest: {primary_manifest}")
    if cqt_manifest is not None:
        print(f"Using cqt manifest: {cqt_manifest}")

    seed = int(tr_cfg.get("seed", 1337))
    seed_everything(seed)

    dataset = UniversalAudioDataset(
        feature_mode=feature_mode,
        manifest_path=primary_manifest,
        class_names=classes,
        cqt_manifest_path=cqt_manifest,
        project_root=str(root),
    )

    if args.dry_run:
        print(f"Resolved dataset with {len(dataset)} samples.")
        if args.resume:
            print(f"Resume checkpoint found: {resume_checkpoint}")
        print("Dry run complete. Config, manifests, classes, and dataset resolved successfully.")
        return

    if len(dataset) < 2:
        raise ValueError(f"Dataset is too small for train/val split: {len(dataset)} samples")

    val_frac = float(tr_cfg.get("val_frac", 0.2))
    val_frac = min(max(val_frac, 0.01), 0.9)
    n_total = len(dataset)
    n_val = max(1, int(round(n_total * val_frac)))
    n_train = n_total - n_val
    if n_train <= 0:
        n_train, n_val = n_total - 1, 1

    split_indices_path = run_dir / "split_indices.pt"
    if args.resume:
        if not split_indices_path.exists():
            raise FileNotFoundError(
                f"Resume requested but split metadata not found: {split_indices_path}"
            )
        train_indices, val_indices = load_saved_split_indices(split_indices_path, n_total)
        train_ds = Subset(dataset, train_indices)
        val_ds = Subset(dataset, val_indices)
        print(
            f"Split: reusing saved split from {split_indices_path} | "
            f"train={len(train_ds)} val={len(val_ds)} total={n_total}"
        )
    else:
        split_gen = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=split_gen)
        print(f"Split: train={len(train_ds)} val={len(val_ds)} total={n_total}")
        torch.save(
            {
                "train_indices": list(train_ds.indices),
                "val_indices": list(val_ds.indices),
                "seed": seed,
                "val_frac": val_frac,
            },
            split_indices_path,
        )

    batch_size = int(tr_cfg.get("batch_size", 32))
    num_workers = int(tr_cfg.get("num_workers", 4))
    use_padding_collate = bool(tr_cfg.get("pad_collate", False))

    device, cuda_amp_default, mps_amp_default, scaler, pin_mem = get_device()
    mixed_precision = bool(tr_cfg.get("mixed_precision", True))
    use_cuda_amp = bool(cuda_amp_default and mixed_precision)
    use_mps_amp = bool(mps_amp_default and mixed_precision)
    if not use_cuda_amp:
        scaler = None
    print(f"Device: {device} | CUDA_AMP={use_cuda_amp} | MPS_AMP={use_mps_amp} | pin_mem={pin_mem}")

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": bool(pin_mem),
    }
    if use_padding_collate:
        loader_kwargs["collate_fn"] = collate_fn_padd

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    print(
        "Dataloaders: "
        f"batch_size={batch_size} | num_workers={num_workers} | pad_collate={use_padding_collate} | "
        f"train_batches={len(train_loader)} | val_batches={len(val_loader)}"
    )

    in_ch = 4 if feature_mode == "mel_cqt" else 2
    backbone = model_cfg.get("backbone", "cnn")
    model = build_model(backbone=backbone, in_ch=in_ch, num_classes=n_classes, model_cfg=model_cfg).to(device)
    model_total_params, model_trainable_params = count_model_parameters(model)
    print(
        "Model: "
        f"backbone={str(backbone).strip().lower()} | in_ch={in_ch} | num_classes={n_classes} | "
        f"dropout={float(model_cfg.get('dropout', 0.3))}"
    )
    print(
        "Model params: "
        f"total={model_total_params:,} | trainable={model_trainable_params:,}"
    )

    pretrained_weights = str(model_cfg.get("pretrained_weights", "")).strip()
    if args.resume and pretrained_weights:
        raise ValueError(
            "Cannot combine --resume with model.pretrained_weights. "
            "Use --resume to continue a stopped run, or pretrained_weights to fine-tune."
        )
    if args.resume:
        print(f"Initialization: resuming full training state from {resume_checkpoint}")
    elif pretrained_weights:
        pretrained_path = resolve_path(pretrained_weights, root)
        if not pretrained_path.exists():
            raise FileNotFoundError(f"pretrained_weights path not found: {pretrained_path}")
        print(f"Initialization: loading fine-tuning weights from {pretrained_path}")
        model = load_pretrained_for_finetuning(model, pretrained_path, n_classes, device)
    else:
        print("Initialization: starting from scratch.")

    lr = float(tr_cfg.get("learning_rate", 1e-3))
    weight_decay = float(tr_cfg.get("weight_decay", 1e-4))
    patience = int(tr_cfg.get("patience", 10))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(1, patience // 3),
    )
    scheduler_patience = max(1, patience // 3)
    print(
        "Optimisation: "
        f"optimizer=AdamW(lr={lr}, weight_decay={weight_decay}) | "
        f"scheduler=ReduceLROnPlateau(mode=min, factor=0.5, patience={scheduler_patience}) | "
        f"early_stop_patience={patience}"
    )

    runtime_cfg = dict(config)
    runtime_cfg["task_mode"] = task_mode
    runtime_cfg["feature_mode"] = feature_mode
    runtime_cfg["dataset"] = dataset_name
    runtime_cfg["resolved"] = {
        "train_config": str(train_cfg_path),
        "audio_config": str(audio_cfg_path),
        "labels_config": str(labels_cfg_path),
        "primary_manifest": str(primary_manifest),
        "cqt_manifest": str(cqt_manifest) if cqt_manifest else "",
        "run_dir": str(run_dir),
        "resume_checkpoint": str(resume_checkpoint) if args.resume else "",
    }
    with open(run_dir / "run_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(runtime_cfg, f, sort_keys=False)
    print(f"Resolved runtime config written to: {run_dir / 'run_config.yaml'}")

    start_epoch = 1
    history = None
    best_val_loss = None
    epochs_no_improve = 0
    if args.resume:
        resumed = load_checkpoint(
            resume_checkpoint,
            device,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        last_epoch = int(resumed.get("epoch", 0))
        start_epoch = last_epoch + 1
        history = resumed.get("history", {}) or None
        best_val_loss = float(resumed.get("best_val_loss", float("inf")))
        if isinstance(history, dict):
            saved_val_history = history.get("val_loss", [])
            if isinstance(saved_val_history, list) and saved_val_history:
                best_val_loss = min(best_val_loss, min(float(v) for v in saved_val_history))
        if "epochs_no_improve" in resumed:
            epochs_no_improve = int(resumed.get("epochs_no_improve", 0))
        else:
            epochs_no_improve = infer_epochs_no_improve(history, best_val_loss)
        print(
            f"Resume state restored: checkpoint epoch={last_epoch}, "
            f"continuing at epoch={start_epoch} | "
            f"best_val_loss={best_val_loss:.6f} | epochs_no_improve={epochs_no_improve}"
        )

    configured_epochs = int(tr_cfg.get("epochs", 50))
    print(
        "Training plan: "
        f"start_epoch={start_epoch} | configured_epochs={configured_epochs} | "
        f"max_new_epochs={max(0, configured_epochs - start_epoch + 1)} | output_dir={run_dir}"
    )
    print("=" * 88)

    trainer = AudioTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        scaler=scaler,
        config=runtime_cfg,
        ckpt_dir=run_dir,
        use_cuda_amp=use_cuda_amp,
        use_mps_amp=use_mps_amp,
    )
    history = trainer.fit(
        train_loader,
        val_loader,
        pin_mem=bool(pin_mem),
        start_epoch=start_epoch,
        history=history,
        best_val_loss=best_val_loss,
        epochs_no_improve=epochs_no_improve,
    )
    write_history_csv(history, run_dir / "history.csv")

    print(f"Training complete. Outputs written to: {run_dir}")
    print(f"Best checkpoint: {run_dir / 'best_val.pt'}")
    print(f"Last checkpoint: {run_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
