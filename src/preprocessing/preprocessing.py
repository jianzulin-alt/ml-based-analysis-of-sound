"""
Disk I/O, waveform manipulation, and caching utilities for audio preprocessing.
"""
import hashlib
import importlib.util
from pathlib import Path
from typing import Optional, Set

import librosa
import numpy as np
import soundfile as sf
import pyloudnorm as pyln

from src.preprocessing.features import compute_stft_params, compute_stereo_logmel_db

# --- File & Path Utilities ---

def ensure_directory_exists(path: Path) -> Path:
    """Create the directory tree if it does not exist and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path

def generate_path_hash(path_str: str) -> str:
    """Generate a short MD5 hash for a file path to prevent filename collisions in cache."""
    return hashlib.md5(path_str.encode("utf-8")).hexdigest()[:10]

def get_valid_labels(cfg: dict) -> Optional[Set[str]]:
    """Extract and sanitise a set of training labels from the configuration dictionary."""
    labels = cfg.get("train_labels") or (cfg.get("dataset") or {}).get("train_labels")
    if not labels:
        return None

    return {str(x).strip().lower() for x in labels if x is not None}

# --- Waveform Manipulation ---

def load_audio_as_stereo_and_resample(path: Path, target_sr: int) -> np.ndarray:
    """
    Load audio, force exactly two channels (stereo), and resample to target rate.
    Returns: (2, Time) numpy array.
    """
    try:
        # soundfile is faster than librosa for initial I/O
        waveform, sr_in = sf.read(str(path), always_2d=True)
    except Exception as e:
        raise ValueError(f"Could not read {path}: {e}")

    # Transpose from (Time, Channels) to (Channels, Time)
    waveform = waveform.T 

    # Resample if the source sample rate differs from the target
    if sr_in != target_sr:
        # Check for resampy to determine the highest quality resampling method available
        _RESAMPLE_TYPE = "kaiser_fast" if importlib.util.find_spec("resampy") else "polyphase"
        waveform = librosa.resample(waveform, orig_sr=sr_in, target_sr=target_sr, res_type=_RESAMPLE_TYPE)

    # Standardise channel count
    channels, _ = waveform.shape
    if channels == 1:
        # Duplicate mono signal to stereo
        stereo = np.vstack([waveform, waveform])
    elif channels == 2:
        stereo = waveform
    else:
        # Truncate multichannel audio to the first two channels
        stereo = waveform[:2, :]

    return stereo.astype(np.float32)

def conform_audio_duration(stereo: np.ndarray, sr: int, duration_s: float) -> np.ndarray:
    """Pad with silence or crop a stereo waveform to an exact duration in seconds."""
    _, current_samples = stereo.shape
    target_samples = int(round(sr * duration_s))
    
    if current_samples >= target_samples:
        return stereo[:, :target_samples]
    
    padding = target_samples - current_samples
    return np.pad(stereo, ((0, 0), (0, padding)), mode='constant')

# --- Loudness Normalisation ---

def apply_lufs_normalisation(
    stereo: np.ndarray,
    sr: int,
    target_lufs: float = -23.0,
    peak_limit: float = 0.99,
) -> np.ndarray:
    """
    Apply EBU R128 loudness normalisation to a stereo waveform.
    Ensures consistent volume across disparate audio sources (e.g., film vs. field recordings).
    """
    if pyln is None:
        raise RuntimeError("pyloudnorm is required for LUFS normalisation.")

    # pyloudnorm expects (Time, Channels)
    waveform_tc = stereo.T
    meter = pyln.Meter(sr)
    
    try:
        measured_lufs = meter.integrated_loudness(waveform_tc)
        if not np.isfinite(measured_lufs):
            return stereo
    except Exception:
        return stereo

    # Calculate gain factor
    gain_db = target_lufs - measured_lufs
    gain = 10.0 ** (gain_db / 20.0)
    normalised_tc = waveform_tc * gain

    # Apply peak limiting to prevent digital clipping
    if peak_limit and peak_limit > 0:
        peak = np.max(np.abs(normalised_tc)) + 1e-12
        if peak > peak_limit:
            normalised_tc *= (peak_limit / peak)

    return normalised_tc.T.astype(np.float32)

def preprocess_loudness(
    stereo: np.ndarray,
    sr: int,
    loudness_norm: str = "none",  
    target_lufs: float = -23.0,
    peak_limit: float = 0.99,
) -> np.ndarray:
    """Orchestrate loudness normalisation based on the specified mode."""
    # Ensure we use the new variable name inside the function
    norm_type = str(loudness_norm or "none").strip().lower()
    
    if norm_type in {"none", "off", "false", "0"}:
        return stereo
    if norm_type == "lufs":
        return apply_lufs_normalisation(stereo, sr, target_lufs, peak_limit)
        
    raise ValueError(f"Unsupported loudness_norm mode: {loudness_norm}")

# --- Caching Pipeline ---

def cache_stereo_logmel(
    wav_path: Path, 
    label: str, 
    cache_root: Path,
    sr: int, 
    dur: float, 
    n_mels: int, 
    win_ms: float, 
    hop_ms: float,
    fmin: float, 
    fmax: Optional[float],
    window: str = "hann",
    loudness_norm: str = "none",
    target_lufs: float = -23.0,
    loudness_peak_limit: float = 0.99
) -> Path:
    """
    Full pipeline: Load -> Conform -> Normalise -> Extract -> Cache (.npy).
    """
    # 1. Prepare DSP Parameters
    n_fft, hop, win_length = compute_stft_params(sr, win_ms, hop_ms)
    
    # 2. Waveform Processing
    stereo = load_audio_as_stereo_and_resample(wav_path, target_sr=sr)
    stereo = conform_audio_duration(stereo, sr, dur)
    stereo = preprocess_loudness(
        stereo, sr=sr, loudness_norm=loudness_norm,
        target_lufs=target_lufs, peak_limit=loudness_peak_limit,
    )
    
    # 3. Feature Extraction
    mel = compute_stereo_logmel_db(
        stereo, sr, n_fft=n_fft, hop=hop, win_length=win_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax, window=window,
    )

    # 4. Filename Generation (Encoding parameters into the filename to prevent stale cache)
    ln_mode = str(loudness_norm or "none").strip().lower()
    ln_tag = f"ln{ln_mode}"
    if ln_mode == "lufs":
        lu_val = str(float(target_lufs)).replace(".", "p").replace("-", "m")
        ln_tag = f"{ln_tag}_lu{lu_val}"
    
    params_tag = f"sr{sr}_dur{dur}_m{n_mels}_w{int(win_ms)}_h{int(hop_ms)}_{window}_{ln_tag}"
    filename = f"{wav_path.stem}__{generate_path_hash(str(wav_path))}__{params_tag}.npy"
    
    # 5. Save to Disk
    out_path = ensure_directory_exists(cache_root / label) / filename
    np.save(out_path, mel.astype(np.float32))
    
    return out_path
