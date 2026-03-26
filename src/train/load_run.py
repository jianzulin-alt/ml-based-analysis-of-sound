from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yaml


def resolve_saved_paths(run_cfg: dict, project_root: Path) -> pd.DataFrame:
    resolved_cfg = run_cfg.get("resolved", {}) if isinstance(run_cfg.get("resolved"), dict) else {}
    rows = []
    for key, value in resolved_cfg.items():
        stored = "" if value is None else str(value)
        if not stored:
            rows.append({"key": key, "stored": stored, "absolute": "", "exists": False})
            continue

        path = Path(stored).expanduser()
        absolute = path.resolve() if path.is_absolute() else (project_root / path).resolve()
        rows.append(
            {
                "key": key,
                "stored": stored,
                "absolute": str(absolute),
                "exists": absolute.exists(),
            }
        )
    return pd.DataFrame(rows)


def metric_pairs(columns) -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []
    for base in ("loss", "acc", "accuracy"):
        train_col = f"train_{base}"
        val_col = f"val_{base}"
        if train_col in columns and val_col in columns:
            pairs.append((base, train_col, val_col))
    return pairs

def find_project_root(cwd: Path, hint: str | None = None) -> Path:
    markers = [Path('src/configs/audio_params.yaml'), Path('.git')]

    if hint:
        hp = Path(hint).expanduser().resolve()
        if hp.exists() and (hp / 'src/configs/audio_params.yaml').exists():
            return hp

    here = cwd.resolve()
    for base in [here, *here.parents]:
        if (base / markers[0]).exists() or (base / markers[1]).exists():
            return base
    return here


def resolve_input_path(path_like: str | Path, project_root: Path) -> tuple[Path, list[Path]]:
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p.resolve(), [p.resolve()]

    candidates = [
        (Path.cwd() / p).resolve(),
        (project_root / p).resolve(),
    ]

    # Preserve order while deduplicating
    uniq = []
    seen = set()
    for c in candidates:
        key = str(c)
        if key not in seen:
            uniq.append(c)
            seen.add(key)

    for c in uniq:
        if c.exists():
            return c, uniq

    return uniq[0], uniq


def resolve_run_and_checkpoint(
    run_input: str | Path,
    preferred_weight: str = 'best_val.pt',
    project_root_hint: str | None = None,
) -> dict[str, Any]:
    project_root = find_project_root(Path.cwd(), project_root_hint)
    resolved_input, tried = resolve_input_path(run_input, project_root)

    if not resolved_input.exists():
        tried_msg = '\n'.join([f'- {x}' for x in tried])
        raise FileNotFoundError(
            f"RUN_INPUT does not exist: {run_input}\n"
            f"Working directory: {Path.cwd()}\n"
            f"Project root guess: {project_root}\n"
            f"Tried:\n{tried_msg}"
        )

    if resolved_input.is_file():
        if resolved_input.suffix != '.pt':
            raise ValueError(f"Expected .pt checkpoint file, got: {resolved_input}")
        run_dir = resolved_input.parent
        ckpt_path = resolved_input
    else:
        run_dir = resolved_input
        preferred = run_dir / preferred_weight
        best = run_dir / 'best_val.pt'
        last = run_dir / 'last.pt'

        if preferred.exists():
            ckpt_path = preferred
        elif best.exists():
            ckpt_path = best
        elif last.exists():
            ckpt_path = last
        else:
            ckpt_path = None

    return {
        'project_root': project_root,
        'resolved_input': resolved_input,
        'run_dir': run_dir,
        'ckpt_path': ckpt_path,
        'run_cfg_path': run_dir / 'run_config.yaml',
        'history_csv_path': run_dir / 'history.csv',
        'split_path': run_dir / 'split_indices.pt',
    }


def load_run_config(path: Path) -> dict:
    if not path.exists() or yaml is None:
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def history_dict_to_df(history: dict) -> pd.DataFrame:
    if not isinstance(history, dict) or not history:
        return pd.DataFrame()

    values = {k: v for k, v in history.items() if isinstance(v, list)}
    if not values:
        return pd.DataFrame()

    max_len = max(len(v) for v in values.values())
    for k, v in values.items():
        if len(v) < max_len:
            values[k] = v + [np.nan] * (max_len - len(v))

    return pd.DataFrame(values)


def ensure_epoch_column(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty:
        return history_df

    normalized = history_df.copy()
    if "epoch" not in normalized.columns:
        normalized.insert(0, "epoch", np.arange(1, len(normalized) + 1))
        return normalized

    missing_epoch = normalized["epoch"].isna()
    if missing_epoch.any():
        generated = pd.Series(np.arange(1, len(normalized) + 1), index=normalized.index)
        normalized.loc[missing_epoch, "epoch"] = generated.loc[missing_epoch]

    return normalized


def load_history(history_csv_path: Path, ckpt_path: Path | None):
    notes: list[str] = []

    if history_csv_path.exists():
        history_df = ensure_epoch_column(pd.read_csv(history_csv_path))
        return history_df, 'history.csv', notes

    notes.append(f'Missing history.csv: {history_csv_path}')

    if ckpt_path is None:
        notes.append('No checkpoint found (best_val.pt / last.pt).')
        return pd.DataFrame(), 'none', notes

    try:
        import torch

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        history_df = ensure_epoch_column(history_dict_to_df(ckpt.get('history', {})))
        if history_df.empty:
            notes.append(f'Checkpoint loaded but history payload empty: {ckpt_path}')
            return history_df, f'checkpoint:{ckpt_path.name}', notes
        return history_df, f'checkpoint:{ckpt_path.name}', notes
    except Exception as e:
        notes.append(f'Failed to read checkpoint history from {ckpt_path}: {e}')
        return pd.DataFrame(), 'none', notes
