# Notes

## Spectrogram Parameters

audio:
  sr: 44100
  duration: 3.0
  n_mels: 128
  win_ms: 30.0
  hop_ms: 10.0
  fmin: 20.0
  fmax: 20000.0  

$$N = \text{sr} \times \frac{\text{win\_ms}}{1,000}$$

$$N = 44,100 \times 0.03 = 1323 \text{ samples}$$

## The Trade-off

Frequency Resolution: A larger window gives you a clearer picture of which pitches are being played (better frequency resolution).

Time Resolution: A smaller window tells you exactly when a sound happened (better temporal resolution).

30ms choice: This is a very standard "sweet spot" for speech and music, balancing the need to hear individual notes without blurring the timing too much