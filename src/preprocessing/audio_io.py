"""
Disk I/O and waveform manipulation utilities for audio preprocessing.
"""
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
import pyloudnorm as pyln

from src.preprocessing.features import compute_stft_params, compute_stereo_logmel_db
from src.utils.system_utils import ensure_directory_exists, generate_path_hash

# --- Waveform Manipulation ---

def load_audio_as_stereo_and_resample(path: Path, target_sr: int) -> np.ndarray:
    """
    Load audio, force exactly two channels (stereo), and resample to target rate.
    Returns: (2, Time) numpy array.
    """
    y, sr = sf.read(str(path))
    
    # Handle mono by duplicating the channel
    if y.ndim == 1:
        y = np.stack([y, y], axis=1)
    # Handle multi-channel by taking first two
    elif y.shape[1] > 2:
        y = y[:, :2]

    # soundfile loads as (Time, Channels). Transpose to (Channels, Time)
    y = y.T

    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr, res_type="kaiser_fast")

    return y.astype(np.float32)

def conform_audio_duration(stereo: np.ndarray, sr: int, target_dur_sec: float) -> np.ndarray:
    """Pad or crop audio to exactly target_dur_sec."""
    target_samples = int(round(sr * target_dur_sec))
    current_samples = stereo.shape[1]

    if current_samples == target_samples:
        return stereo

    if current_samples > target_samples:
        return stereo[:, :target_samples]

    pad_width = target_samples - current_samples
    return np.pad(stereo, ((0, 0), (0, pad_width)), mode="constant")

def preprocess_loudness(
    stereo: np.ndarray, 
    sr: int, 
    loudness_norm: str = "none", 
    target_lufs: float = -23.0, 
    peak_limit: float = 0.99
) -> np.ndarray:
    """Normalises loudness to a target LUFS or applies simple peak normalisation."""
    mode = str(loudness_norm).strip().lower()
    if mode == "none":
        return stereo

    if mode == "peak":
        max_val = np.max(np.abs(stereo))
        if max_val > 0:
            return stereo * (peak_limit / max_val)
        return stereo

    if mode == "lufs":
        meter = pyln.Meter(sr)
        # pyln expects (Time, Channels)
        audio_t = stereo.T
        try:
            current_lufs = meter.integrated_loudness(audio_t)
            # Avoid blowing up absolute silence
            if not np.isinf(current_lufs) and current_lufs > -70.0:
                audio_t = pyln.normalize.loudness(audio_t, current_lufs, target_lufs)
                
            # Hard limit to prevent digital clipping after LUFS boost
            max_val = np.max(np.abs(audio_t))
            if max_val > peak_limit:
                audio_t = audio_t * (peak_limit / max_val)
                
            return audio_t.T.astype(np.float32)
            
        except ValueError:
            # Fallback if audio is too short or silent for LUFS calculation
            return preprocess_loudness(stereo, sr, "peak", peak_limit=peak_limit)

    raise ValueError(f"Unknown loudness_norm mode: {loudness_norm}")

def process_audio_file(
    wav_path: Path,
    cache_dir: Path,
    sr: int = 44100,
    dur: float = 3.0,
    win_ms: float = 30.0,
    hop_ms: float = 10.0,
    n_mels: int = 128,
    fmin: float = 20.0,
    fmax: float = 20000.0,
    window: str = "hann",
    loudness_norm: str = "none",
    target_lufs: float = -23.0,
    loudness_peak_limit: float = 0.99
) -> Path:
    """
    Full pipeline: Load -> Conform -> Normalise -> Extract -> Cache (.npy).
    """
    n_fft, hop, win_length = compute_stft_params(sr, win_ms, hop_ms)
    
    stereo = load_audio_as_stereo_and_resample(wav_path, target_sr=sr)
    stereo = conform_audio_duration(stereo, sr, dur)
    stereo = preprocess_loudness(
        stereo, sr=sr, loudness_norm=loudness_norm,
        target_lufs=target_lufs, peak_limit=loudness_peak_limit,
    )
    
    mel = compute_stereo_logmel_db(
        stereo, sr, n_fft=n_fft, hop=hop, win_length=win_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax, window=window,
    )

    ln_mode = str(loudness_norm or "none").strip().lower()
    ln_tag = f"ln{ln_mode}"
    if ln_mode == "lufs":
        lu_val = str(float(target_lufs)).replace(".", "p").replace("-", "m")
        ln_tag += f"_{lu_val}"

    hsh = generate_path_hash(str(wav_path))
    out_name = f"{wav_path.stem}_{hsh}_sr{sr}_dur{dur}_m{n_mels}_w{int(win_ms)}_{ln_tag}.npy"
    out_path = cache_dir / out_name
    
    ensure_directory_exists(out_path.parent)
    np.save(out_path, mel.astype(np.float32))
    
    return out_path