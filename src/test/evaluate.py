from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.data_loader import UniversalAudioDataset
from src.test.utils import (
    build_detailed_report,
    build_single_label_prediction_table,
    choose_subset_indices,
    prepare_irmas_part1_mel_manifest,
    run_model_predictions,

)
from src.train.run_train import (
    _repo_root,
    build_model,
    choose_classes,
    load_yaml,
    normalize_feature_mode,
    resolve_feature_manifests,
    resolve_path,
)
from src.train.utils import collate_fn_padd, get_device, load_checkpoint, seed_everything


@dataclass
class ResolvedEvaluationRun:
    root: Path
    run_dir: Path
    checkpoint_path: Path
    runtime_cfg: dict
    config_path: Path | None
    audio_cfg_path: Path
    labels_cfg_path: Path
    audio_cfg: dict
    labels_cfg: dict
    dataset_cfg: dict
    classes: list[str]
    task_mode: str
    feature_mode: str
    dataset_name: str
    threshold: float
    tr_cfg: dict
    model_cfg: dict


def resolve_runtime_config(checkpoint_path: Path, explicit_config: str | None, root: Path) -> tuple[dict, Path | None]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config")
    if isinstance(cfg, dict) and cfg:
        return cfg, None

    run_cfg_path = checkpoint_path.parent / "run_config.yaml"
    if run_cfg_path.exists():
        return load_yaml(run_cfg_path), run_cfg_path

    if explicit_config:
        explicit_path = resolve_path(explicit_config, root)
        return load_yaml(explicit_path), explicit_path

    raise ValueError(
        "Could not resolve runtime config. Provide --config or evaluate a checkpoint directory that contains run_config.yaml."
    )


def resolve_manifests_from_runtime(runtime_cfg: dict, dataset_cfg: dict, root: Path, run_dir: Path) -> tuple[Path, Path | None]:
    resolved = runtime_cfg.get("resolved", {}) or {}
    primary = str(resolved.get("primary_manifest", "")).strip()
    cqt = str(resolved.get("cqt_manifest", "")).strip()

    primary_path = resolve_path(primary, root) if primary else None
    cqt_path = resolve_path(cqt, root) if cqt else None
    if primary_path is not None and primary_path.exists():
        return primary_path, cqt_path if cqt_path and cqt_path.exists() else None

    return resolve_feature_manifests(
        feature_mode=runtime_cfg["feature_mode"],
        dataset_name=runtime_cfg["dataset"],
        dataset_cfg=dataset_cfg,
        task_mode=runtime_cfg["task_mode"],
        root=root,
        run_dir=run_dir,
    )


def resolve_checkpoint_and_run_dir(
    root: Path,
    *,
    checkpoint: str = "",
    run_dir: str = "",
) -> tuple[Path, Path]:
    run_dir_path = resolve_path(run_dir, root) if run_dir else None
    checkpoint_path = resolve_path(checkpoint, root) if checkpoint else None
    if checkpoint_path is None:
        if run_dir_path is None:
            raise ValueError("Provide --checkpoint or --run_dir.")
        checkpoint_path = run_dir_path / "best_val.pt"
    if run_dir_path is None:
        run_dir_path = checkpoint_path.parent
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return run_dir_path, checkpoint_path


def resolve_evaluation_run(
    *,
    root: Path | None = None,
    checkpoint: str = "",
    run_dir: str = "",
    explicit_config: str = "",
    audio_config_override: str = "",
    labels_config_override: str = "",
    threshold_override: float | None = None,
) -> ResolvedEvaluationRun:
    root = _repo_root() if root is None else Path(root).resolve()
    run_dir_path, checkpoint_path = resolve_checkpoint_and_run_dir(
        root,
        checkpoint=checkpoint,
        run_dir=run_dir,
    )

    runtime_cfg, config_path = resolve_runtime_config(checkpoint_path, explicit_config or None, root)
    task_mode = str(runtime_cfg.get("task_mode", "single_label")).strip().lower()
    if task_mode not in {"single_label", "multi_label"}:
        raise ValueError(f"Unsupported task_mode: {task_mode}")
    feature_mode = normalize_feature_mode(runtime_cfg.get("feature_mode", "mel"))
    dataset_name = str(runtime_cfg.get("dataset", "irmas")).strip().lower()

    resolved = runtime_cfg.get("resolved", {}) or {}
    default_audio_cfg = resolved.get("audio_config", "src/configs/audio_params.yaml")
    default_labels_cfg = resolved.get("labels_config", "src/configs/labels.yaml")
    audio_cfg_path = resolve_path(audio_config_override or default_audio_cfg, root)
    labels_cfg_path = resolve_path(labels_config_override or default_labels_cfg, root)

    audio_cfg = load_yaml(audio_cfg_path)
    labels_cfg = load_yaml(labels_cfg_path)
    datasets_cfg = audio_cfg.get("datasets", {}) or {}
    if dataset_name not in datasets_cfg:
        raise ValueError(f"Dataset '{dataset_name}' not found in {audio_cfg_path}")
    dataset_cfg = datasets_cfg[dataset_name]

    classes = choose_classes(labels_cfg, dataset_name)
    tr_cfg = runtime_cfg.get("training", {}) or {}
    model_cfg = runtime_cfg.get("model", {}) or {}
    threshold = float(threshold_override) if threshold_override is not None else float(
        (runtime_cfg.get("multi_label", {}) or {}).get("threshold", 0.5)
    )

    return ResolvedEvaluationRun(
        root=root,
        run_dir=run_dir_path,
        checkpoint_path=checkpoint_path,
        runtime_cfg=runtime_cfg,
        config_path=config_path,
        audio_cfg_path=audio_cfg_path,
        labels_cfg_path=labels_cfg_path,
        audio_cfg=audio_cfg,
        labels_cfg=labels_cfg,
        dataset_cfg=dataset_cfg,
        classes=classes,
        task_mode=task_mode,
        feature_mode=feature_mode,
        dataset_name=dataset_name,
        threshold=threshold,
        tr_cfg=tr_cfg,
        model_cfg=model_cfg,
    )


def build_dataset_for_run(
    run: ResolvedEvaluationRun,
    *,
    manifest_path: Path,
    cqt_manifest_path: Path | None = None,
) -> UniversalAudioDataset:
    return UniversalAudioDataset(
        feature_mode=run.feature_mode,
        manifest_path=manifest_path,
        class_names=run.classes,
        cqt_manifest_path=cqt_manifest_path,
        project_root=str(run.root),
    )


def build_evaluation_loader(dataset, run: ResolvedEvaluationRun, *, batch_size: int = 0, num_workers: int = -1):
    batch_size = int(batch_size) if batch_size > 0 else int(run.tr_cfg.get("batch_size", 32))
    num_workers = int(num_workers) if num_workers >= 0 else int(run.tr_cfg.get("num_workers", 4))
    use_padding_collate = bool(run.tr_cfg.get("pad_collate", False))

    device, _, _, _, pin_mem = get_device()
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": bool(pin_mem),
    }
    if use_padding_collate:
        loader_kwargs["collate_fn"] = collate_fn_padd
    loader = DataLoader(dataset, **loader_kwargs)
    return loader, device, bool(pin_mem)


def build_evaluation_model(run: ResolvedEvaluationRun, device: str):
    in_ch = 4 if run.feature_mode == "mel_cqt" else 2
    backbone = run.model_cfg.get("backbone", "cnn")
    model = build_model(
        backbone=backbone,
        in_ch=in_ch,
        num_classes=len(run.classes),
        model_cfg=run.model_cfg,
    ).to(device)
    load_checkpoint(run.checkpoint_path, device, model)
    return model


def evaluate_dataset(
    run: ResolvedEvaluationRun,
    dataset,
    *,
    split_label: str,
    output_dir: str = "",
    batch_size: int = 0,
    num_workers: int = -1,
    metadata_extra: dict | None = None,
    save_artifacts: bool = False,
) -> dict:
    if len(dataset) == 0:
        raise ValueError(f"Selected dataset for split '{split_label}' is empty.")

    loader, device, pin_mem = build_evaluation_loader(
        dataset,
        run,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    model = build_evaluation_model(run, device)

    outputs = run_model_predictions(
        model,
        loader,
        device=device,
        task_mode=run.task_mode,
        threshold=run.threshold,
        pin_mem=pin_mem,
    )
    report = build_detailed_report(
        outputs["y_true"],
        outputs["y_pred"],
        outputs["y_prob"],
        run.classes,
        task_mode=run.task_mode,
        threshold=run.threshold,
    )

    out_dir = resolve_path(output_dir, run.root) if output_dir else run.run_dir / f"evaluation_{split_label}"
    metadata = {
        "checkpoint": str(run.checkpoint_path),
        "run_dir": str(run.run_dir),
        "config": str(run.config_path) if run.config_path else str((run.run_dir / "run_config.yaml") if (run.run_dir / "run_config.yaml").exists() else ""),
        "audio_config": str(run.audio_cfg_path),
        "labels_config": str(run.labels_cfg_path),
        "split": split_label,
        "num_samples": len(dataset),
        "feature_mode": run.feature_mode,
        "task_mode": run.task_mode,
        "dataset": run.dataset_name,
        "threshold": run.threshold,
    }
    if metadata_extra:
        metadata.update(metadata_extra)

    plot_paths: dict[str, Path] = {}

    return {
        "run": run,
        "dataset": dataset,
        "outputs": outputs,
        "report": report,
        "out_dir": out_dir,
        "metadata": metadata,
        "plot_paths": plot_paths,
    }


def evaluate_saved_split(
    run: ResolvedEvaluationRun,
    *,
    split: str = "val",
    output_dir: str = "",
    batch_size: int = 0,
    num_workers: int = -1,
    save_artifacts: bool = False,
) -> dict:
    primary_manifest, cqt_manifest = resolve_manifests_from_runtime(
        run.runtime_cfg,
        run.dataset_cfg,
        run.root,
        run.run_dir,
    )
    full_dataset = build_dataset_for_run(
        run,
        manifest_path=primary_manifest,
        cqt_manifest_path=cqt_manifest,
    )

    seed = int(run.tr_cfg.get("seed", 1337))
    seed_everything(seed)

    split_indices_path = run.run_dir / "split_indices.pt"
    subset_indices = choose_subset_indices(
        len(full_dataset),
        split=split,
        split_indices_path=split_indices_path if split_indices_path.exists() else None,
        val_frac=float(run.tr_cfg.get("val_frac", 0.2)),
        seed=seed,
    )
    eval_dataset = Subset(full_dataset, subset_indices)

    return evaluate_dataset(
        run,
        eval_dataset,
        split_label=split,
        output_dir=output_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        save_artifacts=save_artifacts,
        metadata_extra={
            "primary_manifest": str(primary_manifest),
            "cqt_manifest": str(cqt_manifest) if cqt_manifest else "",
        },
    )


def evaluate_irmas_part1_run(
    model_path: str | Path,
    *,
    test_root: str | Path,
    cache_root: str | Path,
    output_dir: str = "",
    batch_size: int = 0,
    num_workers: int = -1,
    force_rebuild_features: bool = False,
    save_artifacts: bool = False,
) -> dict:
    root = _repo_root()
    model_path = resolve_path(model_path, root)
    run = resolve_evaluation_run(
        root=root,
        checkpoint=str(model_path) if model_path.is_file() else "",
        run_dir=str(model_path) if model_path.is_dir() else "",
    )

    if run.dataset_name != "irmas":
        raise ValueError(f"Expected an IRMAS run, got dataset='{run.dataset_name}'")
    if run.task_mode != "single_label":
        raise NotImplementedError("IRMAS Part1 comparison currently supports single-label runs only.")
    if run.feature_mode != "mel":
        raise NotImplementedError("IRMAS Part1 comparison currently supports mel feature runs only.")

    manifest_path = prepare_irmas_part1_mel_manifest(
        Path(test_root),
        Path(cache_root),
        run.audio_cfg,
        run.classes,
        force_rebuild=force_rebuild_features,
        project_root=run.root,
    )
    dataset = build_dataset_for_run(run, manifest_path=manifest_path)
    result = evaluate_dataset(
        run,
        dataset,
        split_label="irmas_part1",
        output_dir=output_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        save_artifacts=save_artifacts,
        metadata_extra={
            "test_root": str(Path(test_root).resolve()),
            "primary_manifest": str(manifest_path),
            "cqt_manifest": "",
            "manifest_source": "irmas_part1_sidecars",
        },
    )

    samples_df = build_single_label_prediction_table(
        result["outputs"]["y_pred"],
        dataset.df,
        run.classes,
        project_root=run.root,
    )
    summary = dict(result["report"]["summary"])
    summary_row = {
        "model_name": run.run_dir.name,
        "checkpoint_path": str(run.checkpoint_path),
        "run_dir": str(run.run_dir),
        "feature_mode": run.feature_mode,
        "num_classes": len(run.classes),
        "num_samples": len(dataset),
        "num_multi_label_clips": int(
            dataset.df.get("all_labels", pd.Series(dtype=str))
            .fillna("")
            .astype(str)
            .str.contains(r"\|")
            .sum()
        ),
        "acc": float(summary.get("acc", 0.0)),
        "macro_f1": float(summary.get("macro_f1", 0.0)),
        "micro_f1": float(summary.get("micro_f1", 0.0)),
        "top1_any_label_acc": float(samples_df["hit_any_label"].mean()) if len(samples_df) else 0.0,
    }

    result["samples_df"] = samples_df
    result["summary_row"] = summary_row
    if save_artifacts:
        samples_df.to_csv(result["out_dir"] / "sample_predictions.csv", index=False)
    return result


def compare_irmas_part1_runs(
    model_paths: list[str | Path],
    *,
    test_root: str | Path,
    cache_root: str | Path,
    output_root: str | Path = "",
    batch_size: int = 0,
    num_workers: int = -1,
    force_rebuild_features: bool = False,
    save_run_artifacts: bool = False,
) -> dict:
    if not model_paths:
        raise ValueError("model_paths is empty. Provide at least one run dir or checkpoint path.")

    results_by_name: dict[str, dict] = {}
    summary_rows: list[dict] = []
    per_class_frames: list[pd.DataFrame] = []
    reference_classes: list[str] | None = None

    output_root_path = resolve_path(output_root, _repo_root()) if output_root else None
    if output_root_path is not None:
        output_root_path.mkdir(parents=True, exist_ok=True)

    for model_path in model_paths:
        model_path_resolved = resolve_path(model_path, _repo_root())
        run_name = model_path_resolved.parent.name if model_path_resolved.is_file() else model_path_resolved.name
        run_output_dir = str(output_root_path / run_name) if (output_root_path is not None and save_run_artifacts) else ""
        result = evaluate_irmas_part1_run(
            model_path_resolved,
            test_root=test_root,
            cache_root=cache_root,
            output_dir=run_output_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            force_rebuild_features=force_rebuild_features,
            save_artifacts=save_run_artifacts,
        )

        classes = result["run"].classes
        if reference_classes is None:
            reference_classes = classes
        elif classes != reference_classes:
            raise ValueError("All compared runs must use the same class list.")

        model_name = result["summary_row"]["model_name"]
        if model_name in results_by_name:
            suffix = 2
            while f"{model_name}_{suffix}" in results_by_name:
                suffix += 1
            model_name = f"{model_name}_{suffix}"
            result["summary_row"]["model_name"] = model_name

        results_by_name[model_name] = result
        summary_rows.append(result["summary_row"])
        per_class_frames.append(result["report"]["per_class"].assign(model_name=model_name))

    comparison_df = pd.DataFrame(summary_rows).sort_values(
        ["top1_any_label_acc", "acc", "macro_f1"],
        ascending=False,
    ).reset_index(drop=True)
    per_class_comparison_df = pd.concat(per_class_frames, ignore_index=True)

    if output_root_path is not None:
        comparison_df.to_csv(output_root_path / "irmas_part1_comparison.csv", index=False)
        per_class_comparison_df.to_csv(output_root_path / "irmas_part1_per_class_comparison.csv", index=False)

    return {
        "comparison_df": comparison_df,
        "per_class_comparison_df": per_class_comparison_df,
        "results_by_name": results_by_name,
        "output_root": output_root_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint using the same config and split metadata as training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default="", help="Path to a checkpoint file such as best_val.pt.")
    parser.add_argument("--run_dir", default="", help="Training run directory. Uses best_val.pt by default.")
    parser.add_argument("--config", default="", help="Optional fallback training YAML if checkpoint metadata is missing.")
    parser.add_argument("--audio_config", default="", help="Optional audio config override.")
    parser.add_argument("--labels_config", default="", help="Optional labels config override.")
    parser.add_argument("--split", default="val", choices=["train", "val", "full"], help="Which dataset split to evaluate.")
    parser.add_argument("--output_dir", default="", help="Optional directory for evaluation outputs.")
    parser.add_argument("--batch_size", type=int, default=0, help="Optional dataloader batch size override.")
    parser.add_argument("--num_workers", type=int, default=-1, help="Optional dataloader worker override.")
    parser.add_argument("--threshold", type=float, default=-1.0, help="Optional multi-label threshold override.")
    args = parser.parse_args()

    run = resolve_evaluation_run(
        checkpoint=args.checkpoint,
        run_dir=args.run_dir,
        explicit_config=args.config,
        audio_config_override=args.audio_config,
        labels_config_override=args.labels_config,
        threshold_override=args.threshold if args.threshold >= 0 else None,
    )
    result = evaluate_saved_split(
        run,
        split=args.split,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_artifacts=True,
    )

    print(f"Evaluated {result['metadata']['num_samples']} samples from split='{args.split}'.")
    print(f"Outputs written to: {result['out_dir']}")
    print(f"Summary metrics: {result['report']['summary']}")
    print(f"Per-class F1 CSV: {result['out_dir'] / 'per_class_metrics.csv'}")


if __name__ == "__main__":
    main()
