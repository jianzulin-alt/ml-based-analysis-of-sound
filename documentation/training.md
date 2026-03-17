# Training

The unified training entry is:

```bash
src/configs/train_params.yaml
```

Edit that file to choose:

- dataset
- feature combination
- model backbone
- optimiser/training parameters

Then run training with either:

```bash
make train-dry
make train
```

or directly:

```bash
python -m src.train.run_train --config src/configs/train_params.yaml
```

## Current support

- `task_mode`: `single_label` only
- `feature_mode`:
  - `mel`
  - `cqt`
  - `mfcc`
  - `chroma`
  - `mel_cqt`
  - `mel_chroma`
  - `mfcc_cqt_chroma`
- `model.backbone`:
  - `cnn`
  - `cnn_densenet_121`
  - `baseline_multifeature_cnn`
  - `fusion_attention_cnn`

## File layout

- Main editable training config:
  - `src/configs/train_params.yaml`
- Audio/data config:
  - `src/configs/audio_params.yaml`
- Labels config:
  - `src/configs/labels.yaml`
- Example presets:
  - `src/configs/training/irmas/mel_cnn.yaml`
  - `src/configs/training/irmas/mel_densenet_121.yaml`
  - `src/configs/training/chinese_instruments/single_label_mel_cnn.yaml`

## CLI options

```bash
make train-dry TRAIN_CONFIG=src/configs/training/irmas/mel_cnn.yaml
```

Direct CLI equivalent:

```bash
python -m src.train.run_train \
  --config src/configs/training/irmas/mel_cnn.yaml \
  --audio_config src/configs/audio_params.yaml \
  --labels_config src/configs/labels.yaml \
  --output_dir src/models/saved_weights/my_run \
  --dry_run
```

Make variables:

- `TRAIN_CONFIG`: training YAML to use. Default: `src/configs/train_params.yaml`
- `TRAIN_OUTPUT_DIR`: optional output directory override

Run targets:

- `make train-dry`
- `make train`
- `make train-resume`
- `make train-help`

## Resume

Resume must point at the same run directory used for the original training run.

Required files:

- `last.pt`
- `split_indices.pt`

Example:

```bash
make train-resume \
  TRAIN_CONFIG=src/configs/training/irmas/mel_densenet_121.yaml \
  TRAIN_OUTPUT_DIR=src/models/saved_weights/irmas_single_label_mel_densenet_121
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
