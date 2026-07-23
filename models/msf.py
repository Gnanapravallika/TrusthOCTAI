"""Multi-Scale Fusion projection module for combining multi-depth CNN feature outputs."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFusion(nn.Module):
    """Concatenation and channel projection of multi-scale CNN layer outputs."""

    def __init__(self, layer3_channels: int = 1024, layer4_channels: int = 2048, out_channels: int = 512):
        super().__init__()
        in_channels = layer3_channels + layer4_channels
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, layer3_out: torch.Tensor, layer4_out: torch.Tensor) -> torch.Tensor:
        h4, w4 = layer4_out.shape[2], layer4_out.shape[3]
        layer3_resized = F.interpolate(layer3_out, size=(h4, w4), mode="bilinear", align_corners=False)
        fused = torch.cat([layer3_resized, layer4_out], dim=1)
        out = self.proj(fused)
        out = self.bn(out)
        return self.relu(out)
