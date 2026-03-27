from src.preprocessing.feature_modes import feature_mode_to_features
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, classification_report

from src.train.load_run import resolve_run_and_checkpoint, load_history, load_run_config
from src.data_loader import FeatureFusionDataset
from src.preprocessing.feature_modes import feature_mode_to_features, feature_mode_to_in_channels, manifest_suffix_for_feature
from src.utils.system_utils import load_yaml

def resolve_feature_manifests_for_run(run_cfg, project_root):
    resolved_cfg = run_cfg.get("resolved", {}) or {}
    dataset_name = run_cfg.get("dataset")
    feature_mode = run_cfg.get("feature_mode", "mel")
    feature_names = feature_mode_to_features(feature_mode)

    primary_manifest_rel = resolved_cfg.get("primary_manifest")
    saved_feature_manifests = resolved_cfg.get("feature_manifests", {}) or {}
    audio_cfg = None

    if primary_manifest_rel:
        manifest_path = project_root / primary_manifest_rel
    else:
        audio_cfg_rel = resolved_cfg.get("audio_config", "src/configs/audio_params.yaml")
        audio_cfg = load_yaml(project_root / audio_cfg_rel) or {}
        base_manifest_rel = audio_cfg.get("datasets", {}).get(dataset_name, {}).get("manifest")
        if not base_manifest_rel:
            raise KeyError(f"Could not resolve manifest for dataset '{dataset_name}'")
        base_manifest = project_root / base_manifest_rel
        primary_suffix = manifest_suffix_for_feature(feature_names[0])
        manifest_path = base_manifest.with_name(f"{dataset_name}_train_{primary_suffix}.csv")

    feature_manifest_paths = {}
    if len(feature_names) == 1 and not saved_feature_manifests:
        return manifest_path, feature_manifest_paths

    if audio_cfg is None:
        audio_cfg_rel = resolved_cfg.get("audio_config", "src/configs/audio_params.yaml")
        audio_cfg = load_yaml(project_root / audio_cfg_rel) or {}

    base_manifest_rel = audio_cfg.get("datasets", {}).get(dataset_name, {}).get("manifest")
    if not base_manifest_rel:
        raise KeyError(f"Could not resolve manifest set for dataset '{dataset_name}'")
    base_manifest = project_root / base_manifest_rel

    for feature_name in feature_names:
        if feature_name == feature_names[0]:
            feature_manifest_paths[feature_name] = manifest_path
            continue

        saved_path = saved_feature_manifests.get(feature_name)
        if saved_path:
            feature_manifest_paths[feature_name] = project_root / saved_path
            continue

        suffix = manifest_suffix_for_feature(feature_name)
        feature_manifest_paths[feature_name] = base_manifest.with_name(f"{dataset_name}_train_{suffix}.csv")

    return manifest_path, feature_manifest_paths