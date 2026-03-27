# Data

Datasets for training and testing models belong in this directory
To summarise the dataset distribution run the following script: `data/scripts/summarise_data.py`
To generate the dataset from raw audio sources, refer to the scripts in `data/scripts/generate_dataset.py` (Optional)

### Train dataset summary

```plaintext
 strings   1058  (14.8%)  (52m 54s) (3174.0s)
  brass     846  (11.9%)  (42m 18s) (2538.0s)
  woodwind  710  (10.0%)  (35m 30s) (2130.0s)
  sheng     627  (8.8%)  (31m 21s) (1881.0s)
  dizi      620  (8.7%)  (31m) (1860.0s)
  timpani   575  (8.1%)  (28m 45s) (1725.0s)
  erhu      542  (7.6%)  (27m 6s) (1626.0s)
  pipa      495  (6.9%)  (24m 45s) (1485.0s)
  suona     461  (6.5%)  (23m 3s) (1383.0s)
  guzheng   443  (6.2%)  (22m 9s) (1329.0s)
  guqin     282  (4.0%)  (14m 6s) (846.0s)
  xiao      239  (3.4%)  (11m 57s) (717.0s)
  piano     119  (1.7%)  (5m 57s) (357.0s)
  yangqin   116  (1.6%)  (5m 48s) (348.0s)

Total clip counts: 7135
  ```

## A touch of zen film test dataset

Download train dataset from teams under `General/Datasets/train`
Unzip from teams and place in `data/train`

```plaintext
  strings                         45   (12.4%)
  brass                           32   (8.8%)
  sheng                           30   (8.2%)
  woodwind                        30   (8.2%)
  pipa                            29   (8.0%)
  percussion                      28   (7.7%)
  timpani                         26   (7.1%)
  erhu                            23   (6.3%)
  dizi                            12   (3.3%)
  banzi_clapper                   11   (3.0%)
  guqin                           10   (2.7%)
  bell                            8    (2.2%)
  xiao                            8    (2.2%)
  horn                            7    (1.9%)
  qing                            7    (1.9%)
  bass                            5    (1.4%)
  electronic                      5    (1.4%)
  suona                           4    (1.1%)
  triangle                        4    (1.1%)
  voice                           4    (1.1%)
  chanting_scriptures             3    (0.8%)
  gong                            3    (0.8%)
  marimba                         3    (0.8%)
  morin_khuur                     3    (0.8%)
  cymbals                         2    (0.5%)
  drums                           2    (0.5%)
  guzheng                         2    (0.5%)
  operatic_clapper                2    (0.5%)
  operatic_gongs_and_drums        2    (0.5%)
  piano                           2    (0.5%)
  violin                          2    (0.5%)
  bell_toll                       1    (0.3%)
  clapper                         1    (0.3%)
  huiyuans_buddha_light           1    (0.3%)
  operatic_gongs_and_drums_suona  1    (0.3%)
  qing_stone_chime                1    (0.3%)
  snare_drum                      1    (0.3%)
  wind                            1    (0.3%)
  wind_chimes                     1    (0.3%)
  xiao_zen_like                   1    (0.3%)
  yangqin                         1    (0.3%)
```

### Sample Rates

ata/scripts/summarise_wav_sample_rates.py
Root: data/IRMAS/IRMAS-TestingData-Part1
Total wavs: 807
Unique sample rates: 1

Sample rate distribution:

  44100 Hz  807  (100.0%)

--------------------------------------------------
Root: data/test/a-touch-of-zen
Total wavs: 85
Unique sample rates: 1

Sample rate distribution:

  48000 Hz  85  (100.0%)

--------------------------------------------------
Root: data/train
Total wavs: 7135
Unique sample rates: 1

Sample rate distribution:

  44100 Hz  7135  (100.0%)

Combined Summary
Total wavs: 8027
Unique sample rates: 2

Sample rate distribution:

  44100 Hz  7942  (98.9%)
  48000 Hz  85  (1.1%)