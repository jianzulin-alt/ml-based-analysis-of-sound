import argparse
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset, random_split

# Factory and Data imports
from src.models.builder import build_model
from src.data_loader import FeatureFusionDataset
from src.preprocessing.feature_modes import (
    feature_mode_to_features, 
    feature_mode_to_in_channels, 
    manifest_suffix_for_feature,
    normalize_feature_mode
)

# Engine and Utility imports
from src.train.trainer import AudioTrainer
from src.utils.system_utils import get_repo_root, resolve_path, load_yaml, relative_to_root, resolve_run_dir
from src.utils.train_utils import get_device, load_checkpoint, seed_everything, write_history_csv
from src.utils.audio_utils import load_allowed_labels

# ---------------------------------------------------------
# Augmentation & Dataset Wrappers
# ---------------------------------------------------------
class SpecAugment:
    """Randomly masks time and frequency bands on the spectrogram to prevent overfitting."""
    def __init__(self, freq_mask_param: int = 15, time_mask_param: int = 35, p: float = 0.5):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x shape expected: (Channels, Frequencies, Time)
        if torch.rand(1).item() > self.p:
            return x
        
        x_aug = x.clone()
        _, F, T = x_aug.shape
        
        # Frequency Mask
        f = int(torch.randint(0, self.freq_mask_param, (1,)).item())
        f0 = int(torch.randint(0, max(1, F - f), (1,)).item())
        x_aug[:, f0:f0+f, :] = 0
        
        # Time Mask
        t = int(torch.randint(0, self.time_mask_param, (1,)).item())
        t0 = int(torch.randint(0, max(1, T - t), (1,)).item())
        x_aug[:, :, t0:t0+t] = 0
        
        return x_aug

class TransformSubset(Subset):
    """A Subset wrapper that applies a transformation function to the data."""
    def __init__(self, subset: Subset, transform):
        super().__init__(subset.dataset, subset.indices)
        self.transform = transform

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        if self.transform:
            x = self.transform(x)
        return x, y


def resolve_feature_manifests(
    root: Path,
    audio_cfg: dict,
    dataset_name: str,
    feature_mode: str,
) -> tuple[Path, dict[str, Path]]:
    """Resolves the correct CSV manifests based on the chosen feature mode."""
    feature_names = feature_mode_to_features(feature_mode)
    base_manifest_path = root / audio_cfg["datasets"][dataset_name]["manifest"]

    manifest_paths: dict[str, Path] = {}
    for feature_name in feature_names:
        suffix = manifest_suffix_for_feature(feature_name)
        manifest_paths[feature_name] = base_manifest_path.with_name(f"{dataset_name}_train_{suffix}.csv")

    primary_manifest_path = manifest_paths[feature_names[0]]
    return primary_manifest_path, manifest_paths

def main() -> None:
    parser = argparse.ArgumentParser(description="Training entrypoint for Instrument Classification.")
    parser.add_argument("--config", default="src/configs/train_params.yaml", help="Path to training config YAML")
    parser.add_argument("--audio_config", default="src/configs/audio_params.yaml", help="Path to audio config YAML")
    parser.add_argument("--labels_config", default="src/configs/labels.yaml", help="Path to labels config YAML")
    parser.add_argument("--output_dir", help="Override the default output directory for this run")
    parser.add_argument("--resume", action="store_true", help="Resume training from <output_dir>/last.pt")
    parser.add_argument("--dry_run", action="store_true", help="Validate configs, manifests, and dataset wiring, then exit")
    args = parser.parse_args()

    # 1. Setup Paths and Load Configs
    root = get_repo_root()
    train_cfg_path = resolve_path(args.config, root)
    config = load_yaml(train_cfg_path)
    
    audio_cfg_path = resolve_path(args.audio_config, root)
    labels_cfg_path = resolve_path(args.labels_config, root)
    
    audio_cfg = load_yaml(audio_cfg_path)
    
    # 2. Extract Core Parameters
    exp_name = config.get("experiment_name", "exp")
    dataset_name = config.get("dataset", "irmas")
    feature_mode = normalize_feature_mode(config.get("feature_mode", "mel"))
    
    # Create experiment directory
    run_dir = resolve_run_dir(root, args.output_dir, exp_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Resume Safeguards
    # ---------------------------------------------------------
    resume_ckpt_path = run_dir / "last.pt"
    split_indices_path = run_dir / "split_indices.pt"
    if args.resume and not resume_ckpt_path.exists():
        raise FileNotFoundError(f"--resume requested but checkpoint not found: {resume_ckpt_path}")
    if args.resume and not split_indices_path.exists():
        raise FileNotFoundError(f"--resume requested but split file not found: {split_indices_path}")
    
    # Set global seed for reproducibility
    seed_everything(int(config.get("training", {}).get("seed", 1337)))

    # 3. Resolve Manifest Path
    manifest_path, feature_manifest_paths = resolve_feature_manifests(root, audio_cfg, dataset_name, feature_mode)

    # 4. Resolve Classes
    run_config_path = run_dir / "run_config.yaml"

    if args.resume:
        # Lock class order during resume to prevent Label Shuffling Bug
        if not run_config_path.exists():
            raise FileNotFoundError(f"Cannot resume: {run_config_path} is missing.")
        existing_cfg = load_yaml(run_config_path)
        classes = existing_cfg.get("classes", [])
        if not classes:
            raise ValueError("Existing run_config.yaml is missing the 'classes' list.")
        print(f"Resuming: Locked class mapping loaded from {run_config_path.name}")
    else:
        # Fresh Run: Load, sort deterministically, and snapshot
        raw_classes = load_allowed_labels(labels_cfg_path, dataset_name, None) or []
        classes = sorted(list(raw_classes))
        if not classes:
            raise ValueError(f"No classes found for dataset '{dataset_name}' in {labels_cfg_path}")

        # 5. Generate and save run_config.yaml (The Time Capsule)
        resolved_payload = {
            "train_config": relative_to_root(train_cfg_path, root),
            "audio_config": relative_to_root(audio_cfg_path, root),
            "labels_config": relative_to_root(labels_cfg_path, root),
            "primary_manifest": relative_to_root(manifest_path, root),
            "run_dir": relative_to_root(run_dir, root),
            "resume_checkpoint": relative_to_root(resume_ckpt_path, root) if args.resume else "",
        }
        if len(feature_manifest_paths) > 1:
            resolved_payload["feature_manifests"] = {
                name: relative_to_root(path, root) for name, path in feature_manifest_paths.items()
            }

        run_config_payload = {
            "experiment_name": exp_name,
            "task_mode": config.get("task_mode", "single_label"),
            "dataset": dataset_name,
            "feature_mode": feature_mode,
            "classes": classes,
            "model": config.get("model", {}),
            "training": config.get("training", {}),
            "audio_params": audio_cfg.get("audio", {}),  # Critical for consistent evaluation
            "resolved": resolved_payload
        }
        
        with open(run_config_path, "w", encoding="utf-8") as f:
            yaml.dump(run_config_payload, f, sort_keys=False)
        print(f"Run configuration snapshotted to: {run_config_path}")

    # 6. Prepare Data & SpecAugment
    dataset = FeatureFusionDataset(
        feature_mode=feature_mode,
        manifest_path=manifest_path,
        class_names=classes,
        feature_manifest_paths=feature_manifest_paths if len(feature_manifest_paths) > 1 else None,
        project_root=str(root),
    )
    
    val_frac = float(config.get("training", {}).get("val_frac", 0.2))
    
    if args.resume:
        split_data = torch.load(split_indices_path, map_location="cpu", weights_only=False)
        train_indices, val_indices = split_data["train_indices"], split_data["val_indices"]
        print(f"Resuming with saved data split from: {split_indices_path}")
    else:
        train_size = int(len(dataset) * (1 - val_frac))
        val_size = len(dataset) - train_size
        train_ds_temp, val_ds_temp = random_split(dataset, [train_size, val_size])
        train_indices, val_indices = train_ds_temp.indices, val_ds_temp.indices

        torch.save({
            "train_indices": train_indices,
            "val_indices": val_indices
        }, split_indices_path)

    # Apply SpecAugment ONLY to the training subset if configured in YAML
    use_spec_augment = config.get("training", {}).get("use_spec_augment", False)
    if use_spec_augment:
        print("SpecAugment is ENABLED for the training set.")
        
    train_transform = SpecAugment(p=0.5) if use_spec_augment else None
    
    train_ds = TransformSubset(Subset(dataset, train_indices), train_transform)
    val_ds = Subset(dataset, val_indices) # Validation is never augmented

    # ---------------------------------------------------------
    # Dry Run Exit Point
    # ---------------------------------------------------------
    if args.dry_run:
        print(f"\n--- DRY RUN COMPLETE ---")
        print(f"Config: {train_cfg_path}")
        print(f"Run dir: {run_dir}")
        print(f"Dataset size: {len(dataset)} | train={len(train_ds)} | val={len(val_ds)}")
        print(f"SpecAugment Configured: {use_spec_augment}")
        return
    
    # 7. Hardware Setup
    device, cuda_amp, mps_amp, scaler, pin_mem = get_device()
    
    batch_size = int(config.get("training", {}).get("batch_size", 32))
    num_workers = int(config.get("training", {}).get("num_workers", 4))
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=pin_mem
    )

    # 8. Model, Optimizer, Scheduler
    in_ch = feature_mode_to_in_channels(feature_mode)
    model = build_model(
        config["model"]["backbone"], 
        in_ch, 
        len(classes), 
        config["model"]
    ).to(device)
    
    lr = float(config.get("training", {}).get("learning_rate", 0.0003))
    wd = float(config.get("training", {}).get("weight_decay", 0.0001))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # Load State if Resuming
    start_epoch = 1
    history = None
    best_val_loss = None
    epochs_no_improve = 0
    
    if args.resume:
        ckpt = load_checkpoint(
            resume_ckpt_path,
            device=device,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        history = ckpt.get("history", {})
        best_val_loss = ckpt.get("best_val_loss")
        epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))
        print(f"Resumed checkpoint: {resume_ckpt_path} (Starting at epoch: {start_epoch})")

    # 9. Execute Training
    trainer = AudioTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        scaler=scaler,
        config=config,
        ckpt_dir=run_dir,
        use_cuda_amp=cuda_amp,
        use_mps_amp=mps_amp
    )
    
    print(f"\nStarting experiment: '{exp_name}' on {device}")
    history = trainer.fit(
        train_loader,
        val_loader,
        pin_mem=pin_mem,
        start_epoch=start_epoch,
        history=history,
        best_val_loss=best_val_loss,
        epochs_no_improve=epochs_no_improve,
    )
    
    write_history_csv(history, run_dir / "history.csv")
    print(f"\nTraining complete. All artifacts saved to: {run_dir}")

if __name__ == "__main__":
    main()