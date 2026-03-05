# Fusion Mode Procedure

This guide describes the full operation flow for:

- Feature modes: `mel`, `cqt`, `mel_cqt`
- Task modes: `single_label`, `multi_label`
- Auto mixed-data check/generation for Chinese multi-label training

## 1) Environment Setup

```bash
# Windows PowerShell
. .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

## 2) Rebuild Features (Recommended Before New Experiments)

```bash
make clean
make all
```

`make all` generates all current feature presets (with LUFS preprocessing enabled by config).

## 3) Train

### 3.1 Single-label Training

```bash
# Mel
make train TRAIN_DATASET=chinese TRAIN_FEATURE=mel TRAIN_TASK_MODE=single_label

# CQT
make train TRAIN_DATASET=chinese TRAIN_FEATURE=cqt TRAIN_TASK_MODE=single_label

# Mel+CQT
make train TRAIN_DATASET=chinese TRAIN_FEATURE=mel_cqt TRAIN_TASK_MODE=single_label
```

### 3.2 Multi-label Training

```bash
# Mel
make train TRAIN_DATASET=chinese TRAIN_FEATURE=mel TRAIN_TASK_MODE=multi_label AUTO_GENERATE_MIXED=ask

# CQT
make train TRAIN_DATASET=chinese TRAIN_FEATURE=cqt TRAIN_TASK_MODE=multi_label AUTO_GENERATE_MIXED=ask

# Mel+CQT
make train TRAIN_DATASET=chinese TRAIN_FEATURE=mel_cqt TRAIN_TASK_MODE=multi_label AUTO_GENERATE_MIXED=ask
```

Multi-label behavior (Chinese dataset):

1. Training checks whether mixed manifest is ready.
2. If not ready, it asks whether to generate mixed data (when `AUTO_GENERATE_MIXED=ask`).
3. You can force behavior:
   - `AUTO_GENERATE_MIXED=always`
   - `AUTO_GENERATE_MIXED=never`

## 4) Test

```bash
# Single-label CQT example
make test TEST_DATASET=chinese TEST_FEATURE=cqt TEST_TASK_MODE=single_label

# Multi-label Mel+CQT example
make test TEST_DATASET=chinese TEST_FEATURE=mel_cqt TEST_TASK_MODE=multi_label
```

Default test mode:

- `TEST_TASK_MODE=auto` means task mode is inferred from checkpoint metadata.

## 5) Direct CLI (Optional)

### Train CLI

```bash
python src/train/train.py --dataset chinese --feature cqt --task_mode single_label
python src/train/train.py --dataset chinese --feature mel_cqt --task_mode multi_label --auto_generate_mixed ask
```

### Test CLI

```bash
python src/test/test.py --dataset chinese --feature cqt --task_mode single_label
python src/test/test.py --dataset chinese --feature mel_cqt --task_mode multi_label
```

## 6) Switch Locations

### Makefile switches

- `TRAIN_DATASET`
- `TRAIN_FEATURE`
- `TRAIN_TASK_MODE`
- `AUTO_GENERATE_MIXED`
- `TEST_DATASET`
- `TEST_FEATURE`
- `TEST_TASK_MODE`

### Python CLI switches

- `src/train/train.py`
  - `--dataset`
  - `--feature`
  - `--task_mode`
  - `--auto_generate_mixed`
- `src/test/test.py`
  - `--dataset`
  - `--feature`
  - `--task_mode`

## 7) Practical Fast Start

```bash
make clean
make all
make train TRAIN_DATASET=chinese TRAIN_FEATURE=mel_cqt TRAIN_TASK_MODE=multi_label AUTO_GENERATE_MIXED=ask
make test TEST_DATASET=chinese TEST_FEATURE=mel_cqt TEST_TASK_MODE=multi_label
```

## 8) Visualization Outputs

After training, the script automatically saves:

- `training_curves_<feature>_<task_mode>.png`
- `training_history_<feature>_<task_mode>.csv`

Location: `src/models/saved_weights/<run_name>/`

The plot title explicitly shows feature mode:

- `Mel`
- `CQT`
- `Mel + CQT`

After testing, the script automatically saves:

- single-label: `test_eval_single_<feature>.png`
- multi-label: `test_eval_multilabel_<feature>.png`

Location: the checkpoint folder (same folder as `best_val.pt`).
