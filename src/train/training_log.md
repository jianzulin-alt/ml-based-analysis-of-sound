# Training Log

Used to keep track of uncompleted train runs

## Resume training densenet from last checkpoint

```bash
python -m src.train.run_train \
  --config src/configs/training/irmas/single_label_mel_densenet_121.yaml --resume
```

python -m src.train.run_train \
  --config src/configs/training/irmas/single_label_mel.yaml --resume

## Completed