
# Training Log

Used to keep track of active, pending, and completed model training runs.

## TO TRAIN

python -m src.scripts.run_train --config src/configs/training/irmas/mel_densenet.yaml --resume

python -m src.scripts.run_train --config src/configs/training/irmas/fusion_attention_mel_cqt.yaml 

python -m src.scripts.run_train --config src/configs/training/irmas/mel_cnn_multilabel.yaml

python -m src.scripts.run_train --config src/configs/training/irmas/fusion_attn_mel_cqt.yaml

## Completed

Mel single label (IRMAS)

python -m src.scripts.run_train \
  --config src/configs/training/chinese_instruments/single_label_mel_cnn.yaml \
  --output_dir src/models/saved_weights/chinese_single_label_mel_cnn_ft_irmas


# Fine-tune (later once chinese dataset is more stable)

python -m src.scripts.run_train --config src/configs/training/chinese_instruments/finetune_fusion_attn_mel_cqt.yaml
python -m src.scripts.run_train --config src/configs/training/chinese_instruments/finetune_fusion_attn_mel_cqt.yaml

python -m src.scripts.run_train \
  --config src/configs/training/chinese_instruments/single_label_mel_cnn.yaml \
  --output_dir src/models/saved_weights/chinese_single_label_mel_cnn_ft_irmas

