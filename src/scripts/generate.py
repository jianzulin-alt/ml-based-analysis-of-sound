#!/usr/bin/env python3
"""
Unified generation entrypoint.

This script consolidates feature/manifest generation tasks that were previously
spread across multiple scripts and Makefile targets.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def find_repo_root() -> Path:
    root = Path.cwd().resolve()
    while root != root.parent and not (root / "src").exists():
        root = root.parent
    return root


def _py_exec() -> str:
    return sys.executable


def _env(repo_root: Path) -> dict:
    env = os.environ.copy()
    current = env.get("PYTHONPATH", "")
    src_path = str((repo_root / "src").resolve())
    env["PYTHONPATH"] = src_path if not current else f"{src_path}{os.pathsep}{current}"
    return env


def _tasks(repo_root: Path) -> Dict[str, List[str]]:
    return {
        "convert_mp3_wav": [
            "src/scripts/convert_mp3_to_wav.py",
            "--root",
            "data/audio/chinese_instruments",
            "--sr",
            "44100",
            "--channels",
            "2",
        ],
        "chinese_mel": [
            "src/scripts/generate_log_mels.py",
            "--config",
            "src/configs/audio_params.yaml",
            "--labels_file",
            "src/configs/labels.yaml",
            "--num_workers",
            os.environ.get("NUM_WORKERS", "19"),
        ],
        "chinese_cqt": [
            "src/scripts/generate_chinese_train_cqt.py",
            "--config",
            "src/configs/audio_params.yaml",
            "--labels_file",
            "src/configs/labels.yaml",
            "--train_dir",
            "data/train",
            "--cqt_cache_root",
            "data/processed/log_cqt",
            "--num_workers",
            os.environ.get("NUM_WORKERS", "19"),
        ],
        "mixed_mel": [
            "src/scripts/generate_mixed_train_mels.py",
            "--config",
            "src/configs/audio_params.yaml",
            "--labels_file",
            "src/configs/labels.yaml",
            "--train_dir",
            "data/train",
            "--out_cache_root",
            "data/processed/log_mels_mixed",
            "--out_manifest",
            "data/processed/train_mels_mixed.csv",
            "--num_mixes",
            os.environ.get("NUM_MIXES", "12000"),
            "--num_workers",
            os.environ.get("NUM_WORKERS", "19"),
            "--save_wavs",
            "--wav_out_dir",
            "data/processed/debug/mixed_wavs",
            "--max_wavs",
            "50",
        ],
        "mixed_mel_cqt": [
            "src/scripts/generate_mixed_train_mel_cqt.py",
            "--config",
            "src/configs/audio_params.yaml",
            "--labels_file",
            "src/configs/labels.yaml",
            "--train_dir",
            "data/train",
            "--out_cache_root",
            "data/processed/log_mels_mixed",
            "--out_cqt_root",
            "data/processed/log_cqt_mixed",
            "--out_manifest",
            "data/processed/train_mels_mixed.csv",
            "--num_mixes",
            os.environ.get("NUM_MIXES", "12000"),
            "--num_workers",
            os.environ.get("NUM_WORKERS", "19"),
            "--save_wavs",
            "--wav_out_dir",
            "data/processed/debug/mixed_wavs",
            "--max_wavs",
            "50",
        ],
        "irmas_mel": [
            "src/scripts/generate_log_mels.py",
            "--config",
            "src/configs/audio_params_irmas.yaml",
            "--labels_file",
            "src/configs/labels_irmas.yaml",
            "--train_dir",
            "data/audio/IRMAS/IRMAS-TrainingData/IRMAS-TrainingData",
            "--num_workers",
            os.environ.get("NUM_WORKERS", "19"),
        ],
        "irmas_cqt": [
            "src/scripts/generate_irmas_train_cqt.py",
            "--irmas_train_dir",
            "data/audio/IRMAS/IRMAS-TrainingData/IRMAS-TrainingData",
            "--cache_root",
            "data/processed/irmas_cqt",
            "--mel_manifest_out",
            "data/processed/irmas_train_mels.csv",
            "--fmax",
            "20000",
            "--num_workers",
            os.environ.get("NUM_WORKERS", "19"),
        ],
        "test_manifest_az": [
            "src/scripts/generate_test_manifest.py",
            "--test_dir",
            "data/test/a-touch-of-zen",
            "--out_csv",
            "data/test/a-touch-of-zen.csv",
        ],
        "test_manifest_irmas": [
            "src/scripts/generate_test_manifest.py",
            "--test_dir",
            "data/audio/IRMAS/IRMAS-TestingData-Part1/IRMAS-TestingData-Part1/Part1",
            "--out_csv",
            "data/test/IRMAS/IRMAS-TestingData-Part1.csv",
        ],
        "irmas_test_cqt": [
            "src/scripts/generate_irmas_test_cqt.py",
            "--input_dir",
            "data/audio/IRMAS/IRMAS-TestingData-Part1/IRMAS-TestingData-Part1/Part1",
            "--cache_root",
            "data/processed/irmas_cqt_test",
            "--manifest_out",
            "data/test/IRMAS/IRMAS-TestingData-Part1.csv",
            "--project_root",
            ".",
            "--dataset_name",
            "IRMAS",
            "--fmax",
            "20000",
            "--num_workers",
            os.environ.get("NUM_WORKERS", "19"),
        ],
    }


def _menu_choice(task_names: List[str]) -> str:
    print("\nSelect generation task:")
    for idx, name in enumerate(task_names, start=1):
        print(f"  {idx}. {name}")
    raw = input("Enter number: ").strip()
    try:
        picked = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid number input: {raw}") from exc
    if not (1 <= picked <= len(task_names)):
        raise ValueError(f"Selection out of range: {picked}")
    return task_names[picked - 1]


def run_task(task: str, repo_root: Path, tasks: Dict[str, List[str]]) -> int:
    if task == "all":
        order = [
            "convert_mp3_wav",
            "chinese_mel",
            "chinese_cqt",
            "mixed_mel_cqt",
            "irmas_mel",
            "irmas_cqt",
            "test_manifest_az",
            "test_manifest_irmas",
            "irmas_test_cqt",
        ]
        for step in order:
            rc = run_task(step, repo_root, tasks)
            if rc != 0:
                return rc
        return 0

    argv = tasks.get(task)
    if argv is None:
        print(f"[ERROR] Unknown task: {task}")
        return 2

    cmd = [_py_exec(), *argv]
    print(f"[generate] Running task={task}")
    print("[generate] Command:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=repo_root, env=_env(repo_root), check=False)
    return int(proc.returncode)


def main() -> int:
    repo_root = find_repo_root()
    tasks = _tasks(repo_root)
    task_names = sorted(tasks.keys()) + ["all"]

    ap = argparse.ArgumentParser(description="Unified generation dispatcher")
    ap.add_argument(
        "--task",
        default=None,
        help="Task name. If omitted, an interactive numbered menu is shown.",
    )
    args = ap.parse_args()

    task = args.task
    if not task:
        task = _menu_choice(task_names)

    return run_task(task, repo_root, tasks)


if __name__ == "__main__":
    raise SystemExit(main())

