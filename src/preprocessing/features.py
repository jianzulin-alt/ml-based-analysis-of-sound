"""
DSP feature computation
"""

from typing import Tuple, Optional
import librosa
import numpy as np

def _next_power_of_two(n: int) -> int:
    """Return the smallest power of two that is greater than or equal to `n`."""
    return 1 << (n - 1).bit_length()

def compute_stft_params(sr: int, win_ms: float, hop_ms: float) -> Tuple[int, int, int]:
    """
    Convert millisecond-based DSP parameters into sample counts.
    
    This function calculates the window length and hop size required for 
    Short-Time Fourier Transform (STFT) operations based on the sample rate.
    It also determines the optimal N_FFT (next power of two) to ensure 
    efficient FFT computation via the Cooley-Tukey algorithm.

    Args:
        sr: Sampling rate (samples per second).
        win_ms: Analysis window duration in milliseconds.
        hop_ms: Step size (stride) between successive windows in milliseconds.

    Returns:
        n_fft: The size of the FFT window (padded to the next power of 2).
        hop: The number of samples between successive frames.
        win_length: The actual number of samples in each analysis window.
    """
    # win_samples = (samples/sec) * (ms / 1000)
    win_length = int(round(sr * (win_ms / 1000.0)))
    hop = int(round(sr * (hop_ms / 1000.0)))
    
    # n_fft is padded to power of 2 for computational efficiency
    n_fft = _next_power_of_two(win_length)
    return n_fft, hop, win_length

def compute_stereo_logmel_db(
    stereo: np.ndarray, 
    sr: int, 
    n_fft: int, 
    hop: int, 
    win_length: int,
    n_mels: int, 
    fmin: float = 20.0, 
    fmax: Optional[float] = None,
    window: str = "hann"
) -> np.ndarray:
    """
    Compute a log-Mel spectrogram for each stereo channel and stack them.
    
    The resulting features are converted to the decibel (dB) scale, which 
    better represents human auditory perception and improves CNN convergence.
    """
    fmax = fmax or (sr / 2)
    window_name = str(window or "hann")
    feats = []
    
    for ch in range(2):
        # Compute Mel-scaled power spectrogram.
        S = librosa.feature.melspectrogram(
            y=stereo[ch], 
            sr=sr, 
            n_fft=n_fft,
            hop_length=hop, 
            win_length=win_length, 
            window=window_name,
            n_mels=n_mels, 
            fmin=fmin, 
            fmax=fmax, 
            power=2.0, 
            center=True
        )
        # Convert power to decibels relative to the peak power in the signal
        S_db = librosa.power_to_db(S, ref=np.max).astype(np.float32)
        feats.append(S_db)
    
    return np.stack(feats, axis=0)  # Shape: (2, n_mels, Time)

def compute_stereo_cqt_db(
    stereo: np.ndarray,
    sr: int,
    n_bins: int,
    bins_per_octave: int,
    hop_length: int,
    fmin: float,
) -> np.ndarray:
    """
    Compute the Constant-Q Transform (CQT) for each stereo channel and stack them.
    
    Unlike STFT, CQT uses bins geometrically spaced in frequency, making it 
    ideal for musical instrument classification and pitch-based analysis.
    """
    feats = []
    for ch in range(2):
        # Compute the CQT (returns complex-valued coefficients)
        C = librosa.cqt(
            y=stereo[ch],
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
        )
        # Convert magnitude to decibels
        C_mag = np.abs(C)
        C_db = librosa.amplitude_to_db(C_mag, ref=np.max).astype(np.float32)
        feats.append(C_db)
        
    return np.stack(feats, axis=0) # Shape: (2, n_bins, Time)
