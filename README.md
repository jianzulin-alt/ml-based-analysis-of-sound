# ML_based_analysis_of_sound

## Machine Learning-Based Analysis of Music and Sound in Martial Arts Films

[Project tasks](https://github.com/users/hughmancoder/projects/4)

## Setup

Install prequisites on your machine
`git, python3, pip, make`

```bash
# Create virtual environment
python -m venv .venv

# On Linux/Mac:
source .venv/bin/activate   

# On Windows (cmd.exe)
.venv\Scripts\activate.bat

# On Windows (PowerShell)
. .venv\Scripts\Activate.ps1

# Upgrade pip tooling (recommended)
python -m pip install --upgrade pip setuptools wheel

# Install base project dependencies
python -m pip install -r requirements.txt
```

Activate environment (venv) on every terminal

### Optional Dependencies (RVC-style Experiments)

These are optional and only needed for advanced feature switches.
Keep them out of the base environment if you only run Mel/Mel+CQT baselines.

```bash
# 1) Self-supervised embeddings (HuBERT / WavLM)
python -m pip install transformers torchaudio accelerate sentencepiece

# 2) Retrieval-based inference (ANN index)
# Preferred:
python -m pip install faiss-cpu
# Fallback if faiss-cpu is not available on your platform:
python -m pip install hnswlib

# 3) Pitch / F0 features
python -m pip install torchcrepe pyworld
```

## Run the project

refer to the make file for command lines

```bash
make help
```

## Quickstart (recommended flow)

1) Prepare datasets (see `data/README.md`).
2) Generate mel features.
3) (Optional) Generate mixed mel features.
4) (Optional) Generate CQT or Mel+CQT features.
5) Train (run `.py` or `.ipynb`).
6) Test (run `.py` or `.ipynb`).

### Common preprocessing targets

```bash
# Unified generation menu (interactive)
make generate

# Unified generation task (non-interactive)
make generate GENERATE_TASK=chinese_mel
make generate GENERATE_TASK=chinese_cqt
make generate GENERATE_TASK=mixed_mel
make generate GENERATE_TASK=mixed_mel_cqt
make generate GENERATE_TASK=irmas_mel
make generate GENERATE_TASK=irmas_cqt
make generate GENERATE_TASK=test_manifest_az
make generate GENERATE_TASK=test_manifest_irmas
make generate GENERATE_TASK=irmas_test_cqt

# Chinese instruments (mel)
make generate_train_mels

# Mixed mel (multilabel mixes)
make generate_mixed_train_mels

# Chinese instruments (CQT, aligned to existing mel manifest)
make generate_chinese_train_cqt

# IRMAS train/test (mel)
make generate_irmas_train_mels
make test_manifest_irmas

# IRMAS train/test (CQT)
make generate_irmas_train_cqt
make generate_irmas_test_cqt

# Generate all feature presets (default one-shot)
make all

# Optional: only generate features needed by selected train/test modes
make all_selected TRAIN_DATASET=chinese TRAIN_FEATURE=mel_cqt TEST_DATASET=chinese TEST_FEATURE=mel_cqt
```

### Training scripts (CLI)

```bash
# Interactive numeric menu (dataset + feature + task_mode)
python src/train/train.py

# Non-interactive examples
python src/train/train.py --dataset chinese --feature mel --task_mode single_label
python src/train/train.py --dataset chinese --feature cqt --task_mode single_label
python src/train/train.py --dataset chinese --feature mel_cqt --task_mode multi_label
python src/train/train.py --dataset irmas --feature mel --task_mode single_label
python src/train/train.py --dataset irmas --feature cqt --task_mode single_label
python src/train/train.py --dataset irmas --feature mel_cqt --task_mode multi_label

# Multi-label mixed-data policy (Chinese):
# ask (default) | always | never
python src/train/train.py --dataset chinese --feature mel_cqt --task_mode multi_label --auto_generate_mixed ask

# Training output plots (auto-saved):
# src/models/saved_weights/<run_name>/training_curves_<feature>_<task_mode>.png
# src/models/saved_weights/<run_name>/training_history_<feature>_<task_mode>.csv
```

### Training notebooks (UI click-run)

Open and run in VSCode/Jupyter:

- `src/train/train.ipynb`

### Test scripts (CLI)

```bash
# Interactive numeric menu (dataset + feature + task_mode)
python src/test/test.py

# Non-interactive examples
python src/test/test.py --dataset chinese --feature mel --task_mode single_label
python src/test/test.py --dataset chinese --feature cqt --task_mode single_label
python src/test/test.py --dataset chinese --feature mel_cqt --task_mode multi_label
python src/test/test.py --dataset irmas --feature mel --task_mode single_label
python src/test/test.py --dataset irmas --feature cqt --task_mode single_label
python src/test/test.py --dataset irmas --feature mel_cqt --task_mode multi_label

# Test output plots (auto-saved in checkpoint folder):
# single-label: test_eval_single_<feature>.png
# multi-label:  test_eval_multilabel_<feature>.png
```

### Test notebooks (UI click-run)

- `src/test/test.ipynb`

## FAQ: CPU Usage Is High but GPU Usage Is Low

**Short answer**: this is usually a data pipeline bottleneck, not a weak model architecture.  
The GPU is waiting for data prepared on CPU.

Common reasons:
- Mel/CQT extraction is CPU-heavy (`librosa` + `numpy`) and does not use GPU.
- DataLoader reads `.npy` files and performs stacking/normalization on CPU.
- On Windows, `num_workers > 0` can sometimes be slower due to process startup/copy overhead.
- Batch size is too small to fully utilize GPU.

Ways to speed up training:
1) **Increase batch size**
   - Usually improves GPU utilization, but increases VRAM usage.
2) **Precompute features**
   - Generate Mel/CQT before training; avoid online feature extraction in the training loop.
3) **Optimize DataLoader**
   - In `utils_mel_cqt.py` / `utils.py`, try `pin_memory=True` and `persistent_workers=True` (validate carefully on Windows).
4) **Use mixed precision**
   - If supported, enable AMP (`torch.cuda.amp.autocast`) in training.
5) **Adjust model/input scale**
   - Larger models or longer sequences can raise GPU load, but fix I/O and batch settings first.

How to identify the bottleneck:
- GPU usage stays low (for example `<30%`) while CPU is saturated: CPU/I/O bottleneck.
- GPU usage is high and VRAM is near full: model compute bottleneck.
