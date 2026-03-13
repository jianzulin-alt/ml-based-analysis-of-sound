# Training Log

Used to keep track of uncompleted train runs

## Resume training densenet from last checkpoint

```bash
python -m src.train.run_train \
  --config src/configs/training/irmas/single_label_mel_densenet_121.yaml --resume
```

python -m src.train.run_train \
  --config src/configs/training/irmas/single_label_mel.yaml --resume

## Fine-tune Chinese Instruments CNN from IRMAS Mel CNN weights

```bash
python -m src.train.run_train \
  --config src/configs/training/chinese_instruments/single_label_mel_cnn.yaml \
  --output_dir src/models/saved_weights/chinese_single_label_mel_cnn_ft_irmas
```

## Completed
