from __future__ import annotations

"""
MultiFeatureFusionAttentionCNN.py

PyTorch implementation of a multi-feature fusion CNN with hybrid attention,
inspired by the uploaded paper. The model assumes three 2D feature maps are
stacked as channels:
    - MFCC
    - CQT
    - Chroma

Expected input shape:
    (batch_size, 3, 128, 128)

The paper describes:
    - three convolutional blocks
    - max-pooling after each block
    - hybrid attention after each block
    - global average pooling
    - two fully connected layers
    - a LogSoftmax output for multi-class classification

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
    """
    Configuration for the fusion-attention CNN.

    Attributes:
        in_channels:
            Number of input channels. For the paper setup this is 3 because
            MFCC, CQT, and Chroma are stacked together.

        num_classes:
            Number of output classes.

        input_size:
            Spatial size of each feature map. The paper uses 128 x 128.

        attention_reduction:
            Reduction ratio used inside the channel attention MLP.

        fc_hidden_dim:
            Hidden dimension of the first fully connected layer.
            The paper mentions two FC layers but does not fully fix the hidden
            size, so this is a tunable choice.

        dropout:
            Dropout applied before the classifier for a bit of regularisation.
            This is a practical addition because papers often omit the exact
            value while still benefitting from it in real training.
    """

    in_channels: int = 3
    num_classes: int = 8
    input_size: Tuple[int, int] = (128, 128)
    attention_reduction: int = 8
    fc_hidden_dim: int = 256
    dropout: float = 0.30


# -----------------------------------------------------------------------------
# Attention blocks
# -----------------------------------------------------------------------------


class ChannelAttention(nn.Module):
    """
    Channel attention module.

    This follows the common squeeze-and-excitation style idea:
        1. Global average pool each channel to a scalar.
        2. Pass through a tiny bottleneck MLP.
        3. Produce per-channel weights in [0, 1].
        4. Reweight the feature map channels.

    Why this makes sense here:
        Different channels/features inside the CNN may carry different levels
        of importance for instrument classification, so channel attention helps
        the network emphasise the more informative filters.
    """

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
    """
    Spatial attention module.

    Spatial attention asks:
        "Where in the time-frequency image should the network focus?"

    A common implementation is used here:
        1. Compute channel-wise average and max projections.
        2. Concatenate them into a 2-channel map.
        3. Use a convolution to produce a 1-channel attention mask.
        4. Reweight the input tensor spatially.

    This is a practical interpretation of the paper's hybrid attention idea.
    """

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
    """
    Hybrid attention = channel attention followed by spatial attention.

    This is a very natural interpretation of the paper's "hybrid attention"
    module. If later you find the paper specifies a different order or a more
    complex fusion rule, you can swap the logic here without changing the rest
    of the model.
    """

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
    """
    A simple convolutional feature extractor block.

    Block layout:
        Conv2d -> BatchNorm2d -> ReLU

    Notes:
        - BatchNorm is added for training stability.
        - Kernel size 3x3 with padding 1 preserves spatial size before pooling.
        - The paper describes sequential CNN layers but may not list every
          normalisation detail, so this is a standard, sensible choice.
    """

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
    """
    One stage of the model:
        ConvBlock -> HybridAttention -> MaxPool

    Pooling reduces the spatial size by a factor of 2, which matches the usual
    design for 128x128 spectrogram-like inputs.
    """

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
    """
    Baseline CNN without attention.

    This is useful for ablation studies so you can test whether the hybrid
    attention module actually helps.
    """

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
            nn.LogSoftmax(dim=1),
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
    """
    Main model implementing the multi-feature fusion CNN with hybrid attention.

    Input:
        x of shape (B, 3, 128, 128)

    Output:
        log-probabilities of shape (B, num_classes)

    Why this name:
        It reflects the main ideas more clearly than a generic filename:
        multiple input features, fusion by channel stacking, CNN backbone,
        and attention-enhanced processing.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Three feature extraction stages with attention.
        self.stage1 = AttentionConvStage(
            in_channels=config.in_channels,
            out_channels=32,
            reduction=config.attention_reduction,
        )
        self.stage2 = AttentionConvStage(
            in_channels=32,
            out_channels=64,
            reduction=config.attention_reduction,
        )
        self.stage3 = AttentionConvStage(
            in_channels=64,
            out_channels=128,
            reduction=config.attention_reduction,
        )

        # Global average pooling compresses each channel to one scalar.
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Two fully connected layers as described in the paper.
        self.fc1 = nn.Linear(128, config.fc_hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(config.fc_hidden_dim, config.num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the pooled feature vector before classification.

        Useful if you want embeddings for visualisation or downstream analysis.
        """
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        x = self.fc1(features)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x


# -----------------------------------------------------------------------------
# Variant that returns raw logits instead of log-probabilities
# -----------------------------------------------------------------------------


class MultiFeatureFusionAttentionCNNLogits(nn.Module):
    """
    Same architecture, but returns raw logits.

    Use this version with nn.CrossEntropyLoss.

    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.backbone = MultiFeatureFusionAttentionCNN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        x = self.backbone.fc1(features)
        x = F.relu(x, inplace=True)
        x = self.backbone.dropout(x)
        logits = self.backbone.fc2(x)
        return logits


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def count_trainable_parameters(model: nn.Module) -> int:
    """Returns the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    # Example configuration.
    config = ModelConfig(
        in_channels=3,
        num_classes=8,
        input_size=(128, 128),
        attention_reduction=8,
        fc_hidden_dim=256,
        dropout=0.30,
    )

    # Instantiate the model.
    model = MultiFeatureFusionAttentionCNN(config)
    print(model)

    # Dummy input representing stacked MFCC, CQT, and Chroma maps.
    x = torch.randn(4, 3, 128, 128)

    # Forward pass.
    y = model(x)

    print("Input shape: ", x.shape)
    print("Output shape:", y.shape)
    print("Trainable parameters:", count_trainable_parameters(model))

    # Example for the logits variant.
    logits_model = MultiFeatureFusionAttentionCNNLogits(config)
    logits = logits_model(x)
    print("Logits shape:", logits.shape)

    # Typical loss usage:
    #   model returning log-probabilities  -> nn.NLLLoss()
    #   model returning raw logits         -> nn.CrossEntropyLoss()

