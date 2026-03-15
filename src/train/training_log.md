# Training Log

Used to keep track of uncompleted train runs

## Training to finish

```bash
python -m src.train.run_train \
  --config src/configs/training/irmas/mel_densenet_121.yaml \
  --output_dir src/models/saved_weights/irmas_single_label_mel_densenet_121 \
  --resume
```


## Completed

python -m src.train.run_train \
  --config src/configs/training/irmas/single_label_mel.yaml --resume


```bash
python -m src.train.run_train \
  --config src/configs/training/chinese_instruments/single_label_mel_cnn.yaml \
  --output_dir src/models/saved_weights/chinese_single_label_mel_cnn_ft_irmas
```