from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DenseLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        bn_size: int = 4,
        p_drop: float = 0.0,
    ) -> None:
        super().__init__()
        inter_channels = bn_size * growth_rate
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(inter_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.p_drop = float(max(0.0, p_drop))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        if self.p_drop > 0.0:
            out = F.dropout(out, p=self.p_drop, training=self.training)
        return out


class _DenseBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        growth_rate: int,
        bn_size: int = 4,
        p_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        channels = in_channels
        for _ in range(num_layers):
            layer = _DenseLayer(channels, growth_rate=growth_rate, bn_size=bn_size, p_drop=p_drop)
            self.layers.append(layer)
            channels += growth_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            new_feat = layer(torch.cat(features, dim=1))
            features.append(new_feat)
        return torch.cat(features, dim=1)


class _Transition(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(self.relu(self.norm(x)))
        x = self.pool(x)
        return x


class CNN_DenseNet_121(nn.Module):
    """
    DenseNet-121 style classifier for spectrogram inputs.

    Signature is kept close to CRNN for drop-in training usage.
    """

    def __init__(
        self,
        num_classes: int,
        p_drop: float = 0.3,
        in_ch: int = 2,
        freq_bins: int = 128,
        growth_rate: int = 32,
        block_config: tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        compression: float = 0.5,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be > 0, got {num_classes}")
        if in_ch <= 0:
            raise ValueError(f"in_ch must be > 0, got {in_ch}")
        if freq_bins <= 0:
            raise ValueError(f"freq_bins must be > 0, got {freq_bins}")

        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Dense blocks + transitions
        channels = num_init_features
        blocks: list[nn.Module] = []
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                in_channels=channels,
                growth_rate=growth_rate,
                bn_size=bn_size,
                p_drop=0.0,
            )
            blocks.append(block)
            channels = channels + num_layers * growth_rate
            if i != len(block_config) - 1:
                out_channels = int(channels * compression)
                blocks.append(_Transition(channels, out_channels))
                channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.norm_final = nn.BatchNorm2d(channels)

        # Head
        self.dropout = nn.Dropout(float(max(0.0, p_drop)))
        self.classifier = nn.Linear(channels, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = F.relu(self.norm_final(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
