from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

BASE_FEATURES = ("mel", "cqt", "mfcc", "chroma")
FEATURE_MANIFEST_SUFFIXES = {
    "mel": "mels",
    "cqt": "cqt",
    "mfcc": "mfcc",
    "chroma": "chroma",
}
FEATURE_CHANNELS = {name: 2 for name in BASE_FEATURES}
SUPPORTED_FEATURE_MODES = {
    "mel": ("mel",),
    "cqt": ("cqt",),
    "mfcc": ("mfcc",),
    "chroma": ("chroma",),
    "mel_cqt": ("mel", "cqt"),
    "mel_chroma": ("mel", "chroma"),
    "mfcc_cqt_chroma": ("mfcc", "cqt", "chroma"),
}
_CANONICAL_FEATURE_ORDER = ("mel", "mfcc", "cqt", "chroma")


def _tokenize_feature_mode(feature_mode: str) -> list[str]:
    raw = str(feature_mode).strip().lower()
    for old, new in (("+", "_"), ("-", "_"), (",", "_"), (" ", "_")):
        raw = raw.replace(old, new)
    return [token for token in raw.split("_") if token]


def normalize_feature_mode(feature_mode: str) -> str:
    raw = str(feature_mode).strip().lower()
    if raw in SUPPORTED_FEATURE_MODES:
        return raw

    tokens = _tokenize_feature_mode(raw)
    if tokens and len(tokens) == len(set(tokens)) and all(token in BASE_FEATURES for token in tokens):
        canonical = "_".join(feature for feature in _CANONICAL_FEATURE_ORDER if feature in tokens)
        if canonical in SUPPORTED_FEATURE_MODES:
            return canonical

    raise ValueError(
        f"Unsupported feature_mode: {feature_mode}. "
        f"Supported modes: {', '.join(SUPPORTED_FEATURE_MODES)}"
    )


def feature_mode_to_features(feature_mode: str) -> tuple[str, ...]:
    return SUPPORTED_FEATURE_MODES[normalize_feature_mode(feature_mode)]


def feature_mode_to_in_channels(feature_mode: str) -> int:
    return sum(FEATURE_CHANNELS[name] for name in feature_mode_to_features(feature_mode))


def manifest_suffix_for_feature(feature_name: str) -> str:
    if feature_name not in FEATURE_MANIFEST_SUFFIXES:
        raise ValueError(f"Unsupported feature name: {feature_name}")
    return FEATURE_MANIFEST_SUFFIXES[feature_name]


def align_and_stack_feature_tensors(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        raise ValueError("At least one feature tensor is required.")

    if len(tensors) == 1:
        return tensors[0]

    min_width = min(int(tensor.shape[-1]) for tensor in tensors)
    target_height = max(int(tensor.shape[-2]) for tensor in tensors)

    aligned: list[torch.Tensor] = []
    for tensor in tensors:
        if tensor.ndim != 3:
            raise ValueError(f"Expected tensor shape (C, H, W), got: {tuple(tensor.shape)}")

        cropped = tensor[:, :, :min_width]
        if cropped.shape[-2] != target_height:
            cropped = F.interpolate(
                cropped.unsqueeze(0),
                size=(target_height, min_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        aligned.append(cropped)

    return torch.cat(aligned, dim=0)
