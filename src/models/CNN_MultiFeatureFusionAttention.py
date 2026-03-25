from __future__ import annotations

"""
MultiFeatureFusionAttentionCNN.py

PyTorch implementation of a multi-feature fusion CNN with hybrid attention.
The model dynamically handles variable-length inputs (e.g., 128 x 300 for 3s audio)
by leveraging AdaptiveAvgPool2d before the fully connected layers.

Expected input shape:
    (batch_size, channels, freq_bins, time_frames)
    e.g., (B, 3, 128, 300) for MFCC+CQT+Chroma
"""

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class ModelConfig:
    in_channels: int = 3
    num_classes: int = 8
    # Input size is purely documentation now; the network is dimension-agnostic.
    input_size: Tuple[int, int] = (128, 300) 
    attention_reduction: int = 8
    fc_hidden_dim: int = 256
    dropout: float = 0.30

# -----------------------------------------------------------------------------
# Attention blocks
# -----------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.mlp(self.avg_pool(x))
        return x * weights

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_map, max_map], dim=1)
        weights = self.activation(self.conv(pooled))
        return x * weights

class HybridAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# -----------------------------------------------------------------------------
# Core CNN blocks
# -----------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class AttentionConvStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.attention = HybridAttention(out_channels, reduction=reduction)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.attention(x)
        x = self.pool(x)
        return x

# -----------------------------------------------------------------------------
# Baseline model without attention
# -----------------------------------------------------------------------------

class BaselineMultiFeatureCNN(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.features = nn.Sequential(
            ConvBlock(config.in_channels, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, config.fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.fc_hidden_dim, config.num_classes),
            # FIX: Removed LogSoftmax. Now safely outputs raw logits.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

# -----------------------------------------------------------------------------
# Main fusion-attention CNN
# -----------------------------------------------------------------------------

class MultiFeatureFusionAttentionCNN(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.stage1 = AttentionConvStage(config.in_channels, 32, config.attention_reduction)
        self.stage2 = AttentionConvStage(32, 64, config.attention_reduction)
        self.stage3 = AttentionConvStage(64, 128, config.attention_reduction)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(128, config.fc_hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(config.fc_hidden_dim, config.num_classes)
        # FIX: Removed LogSoftmax.

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        x = F.relu(self.fc1(features), inplace=True)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

# Alias to prevent breaking imports in run_train.py
MultiFeatureFusionAttentionCNNLogits = MultiFeatureFusionAttentionCNN

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    config = ModelConfig(
        in_channels=3,
        num_classes=8,
        input_size=(128, 300),
        attention_reduction=8,
        fc_hidden_dim=256,
        dropout=0.30,
    )

    model = MultiFeatureFusionAttentionCNN(config)
    
    # Simulating a batch of 3-second audio features (B, C, F, T)
    x = torch.randn(4, 3, 128, 300)
    y = model(x)

    print("Input shape: ", x.shape)
    print("Output shape:", y.shape) # Should perfectly resolve to (4, 8)
    print("Trainable parameters:", count_trainable_parameters(model))