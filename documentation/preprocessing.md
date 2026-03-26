# Preprocessing Theory and Design Choices

## Spectrogram Parameters

```yaml
audio:
  sr: 44100
  duration: 3.0
  n_mels: 128
  win_ms: 30.0
  hop_ms: 10.0
  fmin: 20.0
  fmax: 20000.0  
```

## The STFT Trade-off

When transforming raw audio into a spectrogram, we face a fundamental physical limit known as the Gabor limit (or time-frequency uncertainty principle). We cannot achieve perfect resolution in both time and frequency simultaneously. 

* **Frequency Resolution (Larger Window):** A larger window gathers more samples, giving the Fast Fourier Transform (FFT) a clearer picture of the exact pitches (harmonics) being played. However, it smears transient sounds (like a drum hit) across time.
* **Time Resolution (Smaller Window):** A smaller window tells you exactly *when* a sound happened, but blurs the frequency representation, making it harder to distinguish between closely related pitches.

**The 30ms Choice:** We calculate the window size in samples using the formula $N = \text{sr} \times \frac{\text{win\_ms}}{1,000}$. For a **44100Hz** sample rate, $44,100 \times 0.03 = 1323 \text{ samples}$. This 30ms duration is a widely accepted "sweet spot" for musical and speech analysis, balancing the need to resolve individual instrumental notes without blurring rhythmic articulations.

## Design Choices Aligned with the Literature

1.  **Decibel Scaling:** Human hearing is logarithmic, not linear. An instrument playing twice as loud physically does not sound twice as loud to us. Converting the Mel power spectrogram to a decibel (dB) scale aligns the data with human auditory perception and compresses the dynamic range, aiding neural network convergence.
2.  **Data Augmentation (Channel Swapping):** To artificially expand the dataset and improve generalisation, we duplicate each stereo training example and swap the left and right audio channels. Because most musical recordings are mixed with distinct left/right channel data, this gives the convolutional layers more varied spatial opportunities to learn from.
3.  **Normalisation:** The final decibel arrays must be normalised before being passed to the CNN to ensure stable gradient descent and prevent large unscaled values from saturating the network's activation functions.
