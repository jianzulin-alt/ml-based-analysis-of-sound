# Makefile usage

This project Makefile wraps feature extraction and mixing commands using:

- `PY=.venv/bin/python`
- `CONFIG=src/configs/audio_params.yaml`

## Core variables

- `DATASET`: dataset key from `audio_params.yaml` (`irmas`, `chinese_instruments`)
- `FEATURE`: `mel`, `cqt`, `mel_cqt`
- `WORKERS`: number of parallel worker processes for extraction
- `MIX_DATASET`: source dataset used to build synthetic polyphonic mixtures
- `MIX_FEATURE`: extracted feature type for mixed samples (`mel`, `cqt`, `mel_cqt`; legacy alias: `both`)
- `MIX_WORKERS`: parallel workers for mix generation/extraction
- `LABELS_CONFIG`: label allowlist YAML used by both `extract_features.py` and `mix_and_extract.py`
- `MIX_NUM_MIXES`: optional override for `mixing.num_mixes` (useful for smoke tests)
- `MIX_SEED`: optional override for `mixing.seed`

## Targets

- `make extract DATASET=irmas FEATURE=mel_cqt WORKERS=12`
  - Runs `src/scripts/extract_features.py` with selected dataset/feature/workers.
  - Uses `LABELS_CONFIG` to filter which label folders are included.
- `make irmas FEATURE=mel_cqt WORKERS=12`
  - Shortcut for IRMAS extraction.
- `make chinese FEATURE=mel WORKERS=8`
  - Shortcut for Chinese Instruments extraction.
- `make extract-help`
  - Shows CLI help for `extract_features.py`.
- `make mix`
  - Runs synthetic mixing + feature extraction.
  - Uses `LABELS_CONFIG` to decide which source labels are allowed into the mixing pool.
  - Example: `make mix MIX_DATASET=chinese_instruments MIX_FEATURE=mel_cqt MIX_WORKERS=8`.
  - Quick test: `make mix MIX_DATASET=irmas MIX_FEATURE=mel MIX_WORKERS=1 MIX_NUM_MIXES=1 MIX_SEED=1337`.
- `make clean`
  - Deletes `data/processed`.

## How `src/configs/labels.yaml` is used

`LABELS_CONFIG` defaults to `src/configs/labels.yaml`.

The extraction and mixing scripts use it as an allowlist:

- For `DATASET=irmas`, they prefer `irmas_labels`.
- For `DATASET=chinese_instruments`, they prefer `train_labels`.

Current label groups are:

- `irmas_labels`
  - `cel`, `cla`, `flu`, `gac`, `gel`, `org`, `pia`, `sax`, `tru`, `vio`, `voi`
- `train_labels`
  - `strings`, `brass`, `woodwind`, `sheng`, `dizi`, `timpani`, `erhu`, `pipa`, `suona`, `guzheng`, `piano`, `guqin`, `xiao`, `yangqin`

Practical effect:

- `make extract` only generates `.npy` features and CSV rows for labels included in the configured label list.
- Any folder under the dataset root whose name is not in the allowlist is skipped.
- `make mix` only samples source clips from allowed labels.

If `LABELS_CONFIG` is missing, or the expected key is not found in the YAML, the scripts fall back to using all label folders they find on disk and print a warning.

You can point the Makefile at a different label file:

```bash
make extract DATASET=irmas FEATURE=mel LABELS_CONFIG=src/configs/labels.yaml
```

## How `src/configs/audio_params.yaml` is used

`extract_features.py` loads `--config` and reads:

- `audio`: DSP and preprocessing settings (`sr`, `duration`, `n_mels`, `n_bins`, `bins_per_octave`, `win_ms`, `hop_ms`, `fmin`, `fmax`, `window`, loudness settings).
- `datasets.<DATASET>`: dataset-specific paths:
  - `train_dir`: source wav files
  - `cache_root`: output `.npy` feature directory
  - `manifest`: base path used to write CSV manifests

During extraction, only files whose parent-folder label is allowed by `LABELS_CONFIG` are written into these manifests.

`mix_and_extract.py` also receives the same config file and uses:

- `datasets.<MIX_DATASET>.train_dir`: source wav pool for sampling stems
- `datasets.<MIX_DATASET>.cache_root`: base output path for mixed `.npy` features (with `_mixed` suffix)
- `datasets.<MIX_DATASET>.manifest`: base name used for mixed manifest filenames
- `mixing`: mixture policy (`num_mixes`, `min_sources`, `max_sources`, `snr_db_min`, `snr_db_max`, `seed`)

Mixed manifests include a `labels` column (pipe-separated labels) for multi-label CNN training.

See [training.md](training.md) for the YAML-driven training entrypoint.
