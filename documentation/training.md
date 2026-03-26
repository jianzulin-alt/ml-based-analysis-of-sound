# Training Pipeline

The unified training entry point is driven by YAML configuration files. Edit the primary configuration file to set up your experiment:
`src/configs/train_params.yaml`

Training Configuration Strategy
Recommendation: Use a new train config for every major experiment.


## Quick Start

You can start training using the provided Make targets:
```bash
make train          # Runs training with default configs
make train-dry      # Validates configs and dataset without starting the loop
```

Or via direct Python CLI (useful for overriding specific configs):
```bash
python -m src.scripts.run_train --config src/configs/train_params.yaml
```

---

## Current Support

- **Task Mode:** `single_label` (Multi-label is currently disabled for this pipeline).
- **Feature Modes:** - Single: `mel`, `cqt`, `mfcc`, `chroma`
  - Fusion: `mel_cqt`, `mel_chroma`, `mfcc_cqt_chroma`
- **Model Backbones:** - `cnn`
  - `cnn_densenet_121`
  - `baseline_multifeature_cnn`
  - `CNN_MultiFeatureFusionAttention` (Primary for Honours report)

---

## Advanced CLI Options

You can override paths and run specific presets from the command line:

```bash
python -m src.scripts.run_train \
  --config src/configs/training/irmas/mel_cnn.yaml \
  --audio_config src/configs/audio_params.yaml \
  --labels_config src/configs/labels.yaml \
  --output_dir src/models/saved_weights/my_custom_run \
  --dry_run
```

- `--output_dir`: Overrides the default save location (`src/models/saved_weights/<experiment_name>`).
- `--dry_run`: Resolves configs, manifests, classes, and datasets, then exits cleanly.
- `--resume`: Resumes a stopped run from `<run_dir>/last.pt`.

---

## Resuming vs. Fine-tuning

**Do not combine `--resume` with `model.pretrained_weights`.** They serve different purposes.

### 1. Resuming an Interrupted Run
To continue a run that stopped unexpectedly, point the script to the original output directory and pass the `--resume` flag. 
*Required files in the directory:* `last.pt` and `split_indices.pt`.

```bash
make train-resume \
  TRAIN_CONFIG=src/configs/training/irmas/mel_densenet_121.yaml \
  TRAIN_OUTPUT_DIR=src/models/saved_weights/irmas_single_label_mel_densenet_121
```

### 2. Fine-tuning (Transfer Learning)
To fine-tune a model (e.g., pre-trained on IRMAS) onto a new dataset (e.g., Chinese Instruments), specify the path to the pretrained weights in your YAML config. The model builder will automatically adapt the final classification head to the new dataset's classes.

```yaml
model:
  pretrained_weights: "src/models/saved_weights/irmas/irmas_fusion/best_val.pt"
  backbone: "CNN_MultiFeatureFusionAttention"
```

---

## Outputs

Each training run automatically generates an isolated experiment folder:
`src/models/saved_weights/<experiment_name>/`

Generated Artifacts:
- `run_config.yaml`: A snapshot of the exact configurations and resolved repo-relative paths used for the run, including the dynamically generated class list.
- `history.csv`: Epoch-by-epoch training and validation metrics.
- `best_val.pt`: Model weights from the epoch with the lowest validation loss.
- `last.pt`: Model weights from the most recently completed epoch.
- `split_indices.pt`: The exact train/validation data split used, ensuring reproducibility for future evaluations.
