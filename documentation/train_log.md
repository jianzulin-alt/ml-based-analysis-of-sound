
# Training Log

Used to keep track of active, pending, and completed model training runs.

## TO TRAIN



python -m src.scripts.run_train --config src/configs/training/irmas/fusion_attention_mel_cqt.yaml 

python -m src.scripts.run_train --config src/configs/training/irmas/mel_cnn_multilabel.yaml

python -m src.scripts.run_train --config src/configs/training/irmas/fusion_attn_mel_cqt.yaml

python -m src.scripts.run_train \
  --config src/configs/training/film_instruments/single_label_mel_cnn_ft_irmas.yaml --resume
          

## Completed

Mel single label (IRMAS)


Mel densenet (IRMAS)
python -m src.scripts.run_train --config src/configs/training/irmas/mel_densenet.yaml --resume

# Fine-tune (later once chinese dataset is more stable)
python -m src.scripts.run_train --config src/configs/training/film_instruments/finetune_fusion_attn_mel_cqt.yaml
python -m src.scripts.run_train --config src/configs/training/film_instruments/finetune_fusion_attn_mel_cqt.yaml




## Experiments

For the Next Iteration (Tomorrow): 1.  Focal Loss: If your trainer.py supports it (we added BCEFocalLoss earlier), use it! It mathematically forces the model to pay 10x more attention to the yangqin errors than the strings errors.
2.  Oversampling: Duplicate the audio files for the bottom 5 classes (guzheng, guqin, xiao, piano, yangqin) during training so the network sees them more often.
   