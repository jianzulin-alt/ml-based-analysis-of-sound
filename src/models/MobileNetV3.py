from __future__ import annotations

import math

import torch
import torch.nn as nn

try:
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "torchvision is required for MobileNetV3. Add torchvision to requirements."
    ) from exc


def _adapt_first_conv(conv: nn.Conv2d, in_ch: int, use_pretrained: bool) -> nn.Conv2d:
    new_conv = nn.Conv2d(
        in_channels=in_ch,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )

    if not use_pretrained:
        return new_conv

    with torch.no_grad():
        old_w = conv.weight
        if in_ch == conv.in_channels:
            new_w = old_w
        elif in_ch == 1:
            new_w = old_w.mean(dim=1, keepdim=True)
        else:
            repeats = math.ceil(in_ch / conv.in_channels)
            new_w = old_w.repeat(1, repeats, 1, 1)[:, :in_ch, :, :]
            new_w *= conv.in_channels / float(in_ch)

        new_conv.weight.copy_(new_w)
        if conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(conv.bias)

    return new_conv


class MobileNetV3Small(nn.Module):
    """
    MobileNetV3-Small adapted for mel-spectrogram classification.
    Input shape: (B, in_ch, 128, W), where in_ch is usually 2.
    """

    def __init__(
        self,
        num_classes: int,
        in_ch: int = 2,
        dropout: float = 0.2,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        # TODO: avoid using imagenet weights
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)

        self.backbone.features[0][0] = _adapt_first_conv(
            self.backbone.features[0][0],
            in_ch=in_ch,
            use_pretrained=pretrained,
        )
        self.backbone.classifier[2].p = dropout
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)

    def freeze_feature_extractor(self) -> None:
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_feature_extractor(self) -> None:
        for param in self.backbone.features.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
