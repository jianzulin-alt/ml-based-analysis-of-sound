# Makefile usage

This project `makefile` currently wraps feature extraction commands via:

- `PY=.venv/bin/python`
- `CONFIG=src/configs/audio_params.yaml`
- `LABELS_CONFIG=src/configs/labels.yaml`

## Core variables

- `DATASET`: dataset key from `audio_params.yaml` (`irmas`, `chinese_instruments`)
- `FEATURE`: feature family for extraction (`mel`, `cqt`)
- `WORKERS`: number of parallel worker processes for extraction
- `LABELS_CONFIG`: label allowlist YAML used by `extract_features.py`

Defaults in `makefile`:

- `FEATURE ?= mel`
- `WORKERS ?= 12`
- `DATASET ?= irmas`

Note: `MIX_DATASET`, `MIX_FEATURE`, and `MIX_WORKERS` are still defined in `makefile`, but the `mix` target is currently commented out.

## Targets

- `make extract DATASET=irmas FEATURE=mel WORKERS=12`
  - Runs `src/scripts/extract_features.py` with selected dataset/feature/workers.
  - Uses `LABELS_CONFIG` to filter which label folders are included.
- `make irmas FEATURE=mel WORKERS=12`
  - IRMAS-focused shortcut (inherits normal `extract` behavior).
- `make chinese FEATURE=mel WORKERS=8`
  - Shortcut that sets `DATASET=chinese_instruments` and runs extraction.
- `make extract-help`
  - Shows CLI help for `extract_features.py`.
- `make clean`
  - Deletes `data/processed`.

## Label filtering (`src/configs/labels.yaml`)

`LABELS_CONFIG` defaults to `src/configs/labels.yaml`.

`extract_features.py` uses this file as an allowlist:

- For `DATASET=irmas`, preferred keys are `irmas_labels`, then `train_labels`.
- For other datasets (including `chinese_instruments`), preferred keys are `train_labels`, then `labels`.

Practical effect:

- `make extract` only generates `.npy` features and CSV rows for allowed labels.
- Any folder under the dataset root whose name is not in the allowlist is skipped.

If `LABELS_CONFIG` is missing, or the expected key is not found in the YAML, extraction falls back to all label folders found on disk and prints a warning.

Example override:

```bash
make extract DATASET=irmas FEATURE=mel LABELS_CONFIG=src/configs/labels.yaml
```

## Audio config (`src/configs/audio_params.yaml`)

`extract_features.py` loads `--config` and reads:

- `audio`: DSP/preprocessing settings (`sr`, `duration`, `n_mels`, `n_bins`, `bins_per_octave`, `win_ms`, `hop_ms`, `fmin`, `fmax`, `window`, loudness settings).
- `datasets.<DATASET>`:
  - `train_dir`: source `.wav` files
  - `cache_root`: output `.npy` feature directory
  - `manifest`: base path used to write CSV manifests

Manifest output names are feature-specific:

- Mel extraction writes `<dataset>_train_mels.csv`
- CQT extraction writes `<dataset>_train_cqt.csv`
