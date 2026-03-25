# Config Layout

## Main training entry

Edit this file for normal training runs:

- `src/configs/train_params.yaml`

This is the single active config file used by default by:

- `make train`
- `make train-dry`
- `make train-resume`
- `python -m src.train.run_train`

## Shared configs

- `src/configs/audio_params.yaml`: dataset paths and audio DSP parameters
- `src/configs/labels.yaml`: class lists

## Presets

Reference presets live under:

- `src/configs/training/irmas/`
- `src/configs/training/chinese_instruments/`

Use a preset directly with:

```bash
make train-dry TRAIN_CONFIG=src/configs/training/irmas/mel_cnn.yaml
```

Supported `feature_mode` values:

- `mel`
- `cqt`
- `mfcc`
- `chroma`
- `mel_cqt`
- `mel_chroma`
- `mfcc_cqt_chroma`

Supported `model.backbone` values:

- `cnn`
- `cnn_densenet_121`
- `baseline_multifeature_cnn`
- `fusion_attention_cnn`
