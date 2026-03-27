# Repository Structure

This document serves two primary purposes:
1. To map the physical layout of the repository.
2. To define the canonical ownership, responsibilities, and data flow for the active machine learning pipeline.


## Current Directory Tree

```text
.
├── README.md
├── makefile
├── requirements.txt
├── setup.py
├── data/
│   ├── IRMAS/              # IRMAS pretraining dataset (raw audio)
│   ├── processed/          # Cached .npy feature tensors and generated CSV manifests
│   ├── raw_sources/        # Source audio and label metadata for Chinese instruments
│   └── test/               # Chinese film audio test set
├── documentation/
│   ├── dataset.md
│   ├── repo_structure.md
│   └── training.md
└── src/
    ├── configs/            # YAML configurations (Strictly data, no logic)
    │   ├── audio_params.yaml
    │   ├── labels.yaml
    │   └── train_params.yaml
    ├── data_loader.py      # UniversalDataset: dynamic manifest merging & multi-hot encoding
    ├── models/
    │   ├── builder.py      # Centralised Factory Pattern for model instantiation
    │   ├── CNN.py
    │   ├── CNN_DenseNet_121.py
    │   └── CNN_MultiFeatureFusionAttention.py
    ├── notebooks/          # Visualisation and exploratory data analysis
    │   ├── analyse_train_run.ipynb
    │   └── evaluate_models.ipynb
    ├── preprocessing/      # Digital Signal Processing (DSP) core
    │   ├── audio_io.py     # Waveform loading, duration conforming, LUFS normalisation
    │   ├── feature_modes.py# Feature routing (e.g., mfcc_cqt_chroma channel stacking)
    │   └── features.py     # librosa transformations (Mel, CQT, MFCC, Chroma)
    ├── scripts/            # High-level pipeline orchestrators (CLI entrypoints)
    │   ├── extract_features.py
    │   ├── run_eval.py     # Universal evaluation script
    │   └── run_train.py    # Training orchestrator
    ├── train/              # Core PyTorch mathematical engines
    │   ├── metrics.py      # sklearn evaluation wrappers (F1, Accuracy)
    │   └── trainer.py      # Core epoch loops, loss calculations, and AMP handling
    └── utils/              # Shared boilerplate and cross-module helpers
        ├── audio_utils.py  # Audio parsing and parameter safety checks
        ├── system_utils.py # Path resolution, hash generation, and YAML loading
        └── train_utils.py  # Checkpointing, device selection, and seed fixing
```

## Canonical Flow


1. **Configuration (`src/configs/`)**: 
   The absolute source of truth for audio extraction parameters, label taxonomies, and training hyperparameters.
2. **DSP Extraction (`src/scripts/extract_features.py` -> `src/preprocessing/`)**: 
   Reads raw `.wav` files, standardises loudness/duration, computes features (Mel/CQT/MFCC), and saves them as cached `.npy` tensors alongside a CSV manifest.
3. **Data Ingestion (`src/data_loader.py`)**: 
   The `UniversalAudioDataset` reads the CSV manifests and dynamically stacks the cached `.npy` tensors into the requested `feature_mode` (e.g., 3-channel MFCC+CQT+Chroma).
4. **Model Instantiation (`src/models/builder.py`)**: 
   The factory dynamically provisions the correct CNN backbone and adjusts the final fully connected layers to match the current dataset's class count.
5. **Training (`src/scripts/run_train.py` -> `src/train/trainer.py`)**: 
   Executes the training loop, applying Automatic Mixed Precision (AMP), calculating loss, and persisting `best_val.pt` checkpoints.
6. **Evaluation (`src/scripts/run_eval.py`)**: 
   Loads the trained weights and runs on-the-fly inference against unseen test directories (IRMAS or Chinese Film sets), generating classification reports.
7. **Visualisation (`src/notebooks/`)**: 
   Consumes the output artifacts (history CSVs, prediction DataFrames) to generate learning curves and confusion matrices for reporting.
