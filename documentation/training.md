# Training

This project now uses a YAML-driven training entrypoint:

```bash
python -m src.train.run_train --config src/configs/<config_name>.yaml
```

## Before training

Training expects feature manifests to already exist in `data/processed`.

Use `mel_cqt` when you want both Mel and CQT features.

For multi-label training, make sure the mixed manifests are real training manifests
## Main config files

- `src/configs/train_params.yaml`
  - General template.
- `src/configs/training/...`
  - Dedicated run presets. This is the preferred place for actual experiments.
- `src/configs/audio_params.yaml`
  - Defines dataset paths and processed manifest roots.
- `src/configs/labels.yaml`
  - Defines class order.
  - `irmas` uses `irmas_labels`.
  - `chinese_instruments` uses `train_labels`.

## Training YAML reference

```yaml
experiment_name: "irmas_pretrain_cnn"
task_mode: "single_label"   # single_label | multi_label
feature_mode: "mel"         # mel | cqt | mel_cqt
dataset: "irmas"            # irmas | chinese_instruments

model:
  backbone: "cnn"           # cnn | densenet121 | cnn_densenet_121
  dropout: 0.3
  pretrained_weights: ""

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  val_frac: 0.2
  patience: 10
  num_workers: 8
  seed: 1337
  mixed_precision: true
  pad_collate: false

multi_label:
  threshold: 0.5
```

### Top-level fields

- `experiment_name`
  Run name used for the default output directory: `src/models/saved_weights/<experiment_name>/`.
  If omitted, the runner falls back to `"exp"`.
- `task_mode`
  Must be `single_label` or `multi_label`.
  This selects the loss function, metrics, and which manifests are preferred.
  `single_label` uses cross-entropy.
  `multi_label` uses BCE-with-logits and prefers mixed manifests.
- `feature_mode`
  Must be `mel`, `cqt`, or `mel_cqt`.
  This controls which manifest files are resolved and how many input channels the model receives.
  `mel` and `cqt` use 2-channel inputs.
  `mel_cqt` concatenates both and uses 4-channel inputs.
- `dataset`
  Dataset key looked up in `src/configs/audio_params.yaml` under `datasets:`.
  This also controls which class list is loaded from `src/configs/labels.yaml`.
  Current built-in values are `irmas` and `chinese_instruments`.

### `model` section

- `model.backbone`
  Model architecture name.
  The current runner supports `cnn`, `cnn_densenet_121`.

- `model.dropout`
  Dropout probability passed into the selected model.
  Higher values add more regularisation but can slow convergence if set too high.
- `model.pretrained_weights`
  Optional checkpoint path for fine-tuning.
  Leave it empty for training from scratch.

### `training` section

- `training.epochs`
  Maximum number of epochs to run.
  Training can stop earlier if early stopping triggers.
- `training.batch_size`
  Batch size for both train and validation dataloaders.
  Increase it for throughput if memory allows; reduce it if you hit OOM errors.
- `training.learning_rate`
  Learning rate passed to the `AdamW` optimizer.
- `training.weight_decay`
  Weight decay passed to `AdamW`.
  This is the main L2-style regularisation term used by the optimizer.
- `training.val_frac`
  Fraction of the dataset reserved for validation.
  The runner clamps this to the range `[0.01, 0.9]`.
  The split is random but repeatable when used with the same `seed`.
- `training.patience`
  Early-stopping patience measured in epochs without validation-loss improvement.
  The learning-rate scheduler also uses this value indirectly with `max(1, patience // 3)`.
- `training.num_workers`
  Number of PyTorch dataloader worker processes.
  Higher values can improve loading throughput, but too many can increase memory usage or startup overhead.
- `training.seed`
  Random seed used for Python, NumPy, PyTorch, and the train/validation split.
  The resolved split indices are saved to `split_indices.pt` so evaluation can replay the exact same split.
- `training.mixed_precision`
  Enables automatic mixed precision when the selected device supports it.
  On CUDA this enables CUDA AMP.
  On Apple Silicon this enables MPS AMP.
  On unsupported devices it has no effect.
- `training.pad_collate`
  Optional boolean, default `false`.
  Enables the custom padding collate function.
  Use this only when your feature tensors have variable time dimensions and need batch-time padding.

### `multi_label` section

- `multi_label.threshold`
  Optional float used only for multi-label metrics and evaluation-time thresholding.
  Default is `0.5`.
  This does not change the BCE training loss itself; it only affects how probabilities are converted to predicted labels for reporting.

## How manifest resolution works

`run_train.py` resolves manifests automatically from `audio_params.yaml`.

For single-label training it prefers standard manifests:

- Mel: `*_train_mels.csv`
- CQT: `*_train_cqt.csv`

For multi-label training it prefers mixed manifests first, then falls back:

- Mel: `*_train_mels_mixed.csv`
- CQT: `*_train_cqt_mixed.csv`

For `mel_cqt`, both Mel and CQT manifests must exist.

## Common commands

Validate a specific config without training:

```bash
python -m src.train.run_train \
  --config src/configs/training/irmas/single_label_mel.yaml \
  --dry_run
```

### What `--dry_run` does

`--dry_run` is a setup check. It resolves the training YAML, audio config, labels config, class list, and manifest paths, then builds the dataset and prints the resolved sample count.

Run training:

```bash
python -m src.train.run_train \
  --config src/configs/training/irmas/single_label_mel.yaml
```

Write checkpoints to a custom directory:

```bash
python -m src.train.run_train \
  --config src/configs/training/irmas/single_label_mel.yaml \
  --output_dir src/models/saved_weights/my_run
```

## Example preset files names to store training parameters

- `src/configs/training/irmas/single_label_mel.yaml`
- `src/configs/training/irmas/single_label_cqt.yaml`
- `src/configs/training/irmas/single_label_mel_cqt.yaml`
- `src/configs/training/irmas/multi_label_mel.yaml`
- `src/configs/training/irmas/multi_label_cqt.yaml`
- `src/configs/training/irmas/multi_label_mel_cqt.yaml`
- `src/configs/training/chinese_instruments/single_label_mel.yaml`
- `src/configs/training/chinese_instruments/single_label_cqt.yaml`
- `src/configs/training/chinese_instruments/single_label_mel_cqt.yaml`
- `src/configs/training/chinese_instruments/multi_label_mel.yaml`
- `src/configs/training/chinese_instruments/multi_label_cqt.yaml`
- `src/configs/training/chinese_instruments/multi_label_mel_cqt.yaml`

## Example workflows

IRMAS single-label pretraining:

```bash
python -m src.train.run_train \
  --config src/configs/training/irmas/single_label_mel.yaml
```

Chinese multi-label training on mixed Mel+CQT:

```bash
python -m src.train.run_train \
  --config src/configs/training/chinese_instruments/multi_label_mel_cqt.yaml
```

Fine-tune from a previous checkpoint:

Edit the target config and set:

```yaml
model:
  pretrained_weights: "src/models/saved_weights/irmas_single_label_mel/best_val.pt"
```

The loader restores only matching layers and leaves the classifier head sized for the current dataset.

Resume an interrupted run from the auto-written `last.pt` checkpoint:

```bash
python -m src.train.run_train \
  --config src/configs/training/irmas/single_label_mel.yaml \
  --resume
```

`--resume` continues from the resolved run directory, restores model, optimizer, scheduler, scaler, history, and resumes at the next epoch.
This is different from `model.pretrained_weights`, which only initializes matching model layers for fine-tuning.

## Outputs

Each run writes to:

`src/models/saved_weights/<experiment_name>/`

Files written there:

- `run_config.yaml`
- `history.csv`
- `best_val.pt`
- `last.pt`
- `tmp_manifests/` when manifest adaptation is needed
- `split_indices.pt` so `src.test.evaluate` can replay the exact train/val split

## Evaluation

Use the saved checkpoint plus the run directory metadata:

```bash
python -m src.test.evaluate \
  --run_dir src/models/saved_weights/irmas_single_label_mel
```

Testing reuses the training config and metrics, then adds:

- per-class F1 output in `per_class_metrics.csv`
- confusion matrix plots
- ROC curve plots

## Quick use pattern

1. Generate features and manifests.
2. Pick the closest preset in `src/configs/training/`.
3. Copy or edit that preset for your run.
4. Run `--dry_run`.
5. Start training.
