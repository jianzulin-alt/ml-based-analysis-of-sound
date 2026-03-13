# Training

Training is run through:

```bash
python -m src.train.run_train --config <path-to-training-yaml>
```

## Current support

- `task_mode`: `single_label` only
- `feature_mode`: `mel` or `cqt`
- `model.backbone`: `cnn` or `cnn_densenet_121`


## Required config files

- `src/configs/audio_params.yaml`
  - Dataset roots and manifest bases under `datasets:`.
- `src/configs/labels.yaml`
  - Class lists (`irmas_labels` for IRMAS, `train_labels` otherwise).
- Training run config (pick one of these presets or create your own):
  - `src/configs/training/irmas/single_label_mel.yaml`
  - `src/configs/training/irmas/single_label_mel_densenet_121.yaml`
  - `src/configs/training/chinese_instruments/single_label_mel.yaml`

Note: `run_train.py` has a default `--config` value of `src/configs/train_params.yaml`, but that file is not present in this repo. Pass `--config` explicitly.

## Training YAML reference

```yaml
experiment_name: "irmas_single_label_mel"
task_mode: "single_label"
feature_mode: "mel" # mel | cqt
dataset: "irmas"    # irmas | chinese_instruments

model:
  backbone: "cnn"   # cnn | cnn_densenet_121
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
  pad_collate: false # optional
```

## Manifest resolution

`run_train.py` resolves manifests from `audio_params.yaml` and looks for:

- Mel: `*_train_mels.csv`
- CQT: `*_train_cqt.csv`

Mixed manifests (`*_mixed.csv`) are not used right now.

## Common commands

Dry run:

```bash
python -m src.train.run_train \
  --config src/configs/training/irmas/single_label_mel.yaml \
  --dry_run
```

Train:

```bash
python -m src.train.run_train \
  --config src/configs/training/irmas/single_label_mel.yaml
```

Custom output directory:

```bash
python -m src.train.run_train \
  --config src/configs/training/irmas/single_label_mel.yaml \
  --output_dir src/models/saved_weights/my_run
```

Resume:

```bash
python -m src.train.run_train \
  --config src/configs/training/irmas/single_label_mel.yaml \
  --resume
```

Fine-tune from checkpoint (in YAML):

```yaml
model:
  pretrained_weights: "src/models/saved_weights/irmas_single_label_mel/best_val.pt"
```

Do not combine `--resume` with `model.pretrained_weights`.

## Outputs

Each run writes to:

`src/models/saved_weights/<experiment_name>/`

Files:

- `run_config.yaml`
- `history.csv`
- `best_val.pt`
- `last.pt`
- `split_indices.pt`
- `tmp_manifests/` (only when manifest columns are adapted)

## Quick workflow

1. Prepare features/manifests in `data/processed`.
2. Pick a preset in `src/configs/training/`.
3. Run `--dry_run`.
4. Start training.
