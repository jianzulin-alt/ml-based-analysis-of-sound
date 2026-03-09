## Training

## Recommendations (deep research)

The recommended model architecture is a fine-tuned PANNs (CNN14) or AST, using a stacked CQT-MFCC-Chroma feature set. This backbone should be trained using Asymmetric Loss (ASL) to overcome the dataset's inherent imbalance. To address the unique interference patterns of martial arts soundtracks, the pipeline should include a DME-aware source separation front-end to isolate the music stem before classification. Finally, the use of intensive data augmentation—specifically mixup and impulse response convolution—will ensure the model generalizes from the laboratory environment to the complex, noisy reality of cinematic soundscapes. By prioritizing Macro-F1 and mAP as performance indicators, the project will demonstrate a sophisticated engineering understanding of both signal processing and cultural informatics.

### Pre-training and Domain Adaptation

A robust strategy involves a two-step training pipeline:

Pre-training: Use monophonic, clean datasets of Chinese instruments (like ChMusic or CTIS) to pre-train the model's feature extractor. This allows the model to learn "ideal" instrumental timbres.   

Fine-tuning: Fine-tune the model using the polyphonic film soundtrack dataset. During this phase, data augmentation should be used to bridge the "domain gap" between the clean pre-training data and the noisy film audio

### Features

MFCCs (13 coefficients) for global timbre.

CQT (84 bins) for logarithmic frequency resolution, capturing traditional Chinese pentatonic scales.

Chroma (12 scales) for harmonic content.

## Models

MobileNetV3 (more lightweight)
CNN14	Moderate	Standard SOTA	
General instrument tagging 
Hybrid spatial attention



ResNet38/54	High	Deep feature extraction	Complex polyphonic mixtures

Wavegram-Logmel-CNN	High	Hybrid input	
Capturing both waveform transients and spectral harmonics 

## Try

- Loss Function: Use Asymmetric Loss (ASL) instead of standard Cross-Entropy. ASL allows you to down-weight the millions of "easy" negative samples (where an instrument is not playing) so the model can focus on the rare positive samples like the yangqin.
- Augmentation: Use Mixup. Since you have limited data for some instruments, mixing a dizi clip with a brass clip and combining their labels is an "expensive-looking" but computationally "cheap" way to teach the model how to handle polyphony.