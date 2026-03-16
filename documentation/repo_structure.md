# Repository Structure

## Canonical flow

The current mainline pipeline in this repository is:

1. `src/configs/`
   Source of truth for audio, label, and training configuration.
2. `src/preprocessing/`
   Pure audio I/O and DSP helpers.
3. `src/scripts/extract_features.py`
   Offline feature extraction from raw `.wav` files into cached `.npy` tensors and CSV manifests.
4. `src/data_loader.py`
   Manifest-backed dataset loader used by training.
5. `src/train/`
   Training entrypoint, checkpointing, metrics, and training loop.
6. `src/test/test.py`
   Evaluation and run comparison against raw test audio.

If new functionality overlaps one of these stages, extend the existing stage instead of creating a second implementation.

## Directory ownership

- `src/configs/`: YAML configs only.
- `src/preprocessing/`: waveform loading, resampling, duration handling, loudness handling, DSP feature computation.
- `src/scripts/`: operational scripts that call preprocessing and write manifests.
- `src/data_loader.py`: dataset assembly from manifests and cached features.
- `src/train/`: train-time orchestration and reusable training utilities.
- `src/test/`: evaluation-only logic and reporting.
- `src/models/`: model definitions and saved run artifacts.
- `src/gui/` and `src/inference/`: user-facing inference surfaces. These should depend on shared runtime helpers, not duplicate preprocessing or model-loading logic.
- `data/scripts/`: dataset maintenance utilities only.
- `documentation/legacy/`: historical workflows that are not part of the active pipeline.

## Known duplication hotspots

These areas already show logic drift and should be consolidated before new features are added there:

1. `src/train/run_train.py` and `src/test/test.py`
   Both define their own versions of repo-root resolution, path resolution, YAML loading, class selection, model construction, and device selection.
2. `src/gui/predictor.py` and `src/inference/utils.py`
   Both represent older inference paths and do not match the current preprocessing API.
3. `src/scripts/mix_and_extract.py`
   This is a commented-out alternate extraction pipeline and should be treated as legacy, not as an implementation source.
4. `src/configs/keys.py`
   Holds constants that are redefined elsewhere instead of being imported.

## Rules for future changes

1. Shared path, config, label, device, and model-factory logic should live in one reusable module before adding new train/test/inference features.
2. Preprocessing should have one source of truth under `src/preprocessing/`. GUI, CLI inference, and evaluation should call that layer directly.
3. New run artifact loading logic should be added once and reused by training, testing, and inference.
4. Historical or paused experiments should move to `documentation/legacy/` or an `archive/` area instead of staying beside active code as commented-out scripts.
5. If a new feature needs raw-waveform inference, build it on top of the existing preprocessing functions instead of introducing new helpers with near-identical names.

## Recommended consolidation targets

- Create a shared runtime module for:
  - repo root and path resolution
  - YAML loading
  - class resolution from labels and run configs
  - model factory
  - checkpoint and run artifact discovery
- Split `src/test/test.py` into smaller modules if evaluation work grows further.
- Either repair `src/gui/` and `src/inference/` to match the active pipeline or move them into legacy status until they are updated.
