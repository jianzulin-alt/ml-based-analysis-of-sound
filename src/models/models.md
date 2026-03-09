# Model Comparison Log

## Tracking Format

Use both:
1. `src/models/experiments.csv` for one-row-per-run structured tracking (auto-written by `src/train/train.py`).
2. This file for concise notes and interpretation.
3. `src/models/test_results.csv` for one-row-per-test structured tracking (auto-written by `src/test/test.py`).

Each run directory should contain:
- `run_config.yaml` (all hyperparameters and data sources)
- `summary.json` (best/final metrics)
- `history.csv` (epoch-by-epoch curves)
- `best_val.pt` and `last.pt` (weights)

## Metrics To Compare

- `val_micro_f1` for overall tagging quality
- `val_macro_f1` for class balance / rare instruments
- `val_exact_match` for strict multi-label correctness

## Run Table

| Run Name | Model | Pretrained | Data | Best Val Micro F1 | Best Val Macro F1 | Notes |
|---|---|---|---|---:|---:|---|
| `CNN_v0` | `cnn` | no | `train_mels.csv` | `0.3122` | `0.2053` | Baseline from log-mel only. |
| `MobileNetV3_v1` | `mobilenet_v3_small` | yes | `train_mels.csv + train_mels_mixed.csv` | `TBD` | `TBD` | Freeze backbone for first epochs, then full fine-tune. |

## Legacy Notes

Baseline notes:
- Weights: `src/models/saved_weights/CNN_v0`
- Features: `log-mel`
- Hamming accuracy (label-wise): `59.92%`
- Subset accuracy (exact match): `0.00%`
