# Training

Run training with:

```bash
python -m src.train.run_train --config <training-yaml>
```

## Current support

- `task_mode`: `single_label` only
- `feature_mode`: `mel` or `cqt`
- `model.backbone`: `cnn` or `cnn_densenet_121`

## Required files

- `src/configs/audio_params.yaml`
- `src/configs/labels.yaml`
- a training config under `src/configs/training/`

Current presets in this repo:

- `src/configs/training/irmas/mel_cnn.yaml`
- `src/configs/training/irmas/mel_densenet_121.yaml`
- `src/configs/training/chinese_instruments/single_label_mel_cnn.yaml`

`run_train.py` defaults `--config` to `src/configs/train_params.yaml`, but that file is not present here, so pass `--config` explicitly.

## CLI options

```bash
python -m src.train.run_train \
  --config src/configs/training/irmas/mel_cnn.yaml \
  --audio_config src/configs/audio_params.yaml \
  --labels_config src/configs/labels.yaml \
  --output_dir src/models/saved_weights/my_run \
  --dry_run
```

- `--output_dir`: override the run directory. Default is `src/models/saved_weights/<experiment_name>`
- `--dry_run`: resolve configs, manifests, classes, and dataset, then exit
- `--resume`: resume from `<run_dir>/last.pt` and `<run_dir>/split_indices.pt`

## Resume

Resume must point at the same run directory used for the original training run.

Required files:

- `last.pt`
- `split_indices.pt`

Example:

```bash
python -m src.train.run_train \
  --config src/configs/training/irmas/mel_densenet_121.yaml \
  --output_dir src/models/saved_weights/irmas_single_label_mel_densenet_121 \
  --resume
```

## Fine-tuning

Set pretrained weights in the YAML:

```yaml
model:
  pretrained_weights: "src/models/saved_weights/irmas/irmas_single_label_mel_cnn/best_val.pt"
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

`run_config.yaml` stores the resolved config paths as repo-relative paths and includes the class list used for the run.
