"""Dual Convolutional Block Attention Module (CBAM) implementation.

Sequentially applies Channel Attention ("WHAT" features matter) and Spatial Attention ("WHERE" lesions exist).
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel Attention Sub-Module using Global Avg & Max Pooling + Shared MLP."""

    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP with reduction ratio r=16
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Sub-Module using Channel Pooling + 7x7 Convolution."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(concat))


class CBAM(nn.Module):
    """Convolutional Block Attention Module combining Channel and Spatial Attention."""

    def __init__(self, gate_channels: int = 2048, reduction_ratio: int = 16, spatial_kernel_size: int = 7):
        super().__init__()
        # Dual names for backwards compatibility with all state dict key patterns
        self.channel_attention = ChannelAttention(gate_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

        # Aliases for shorter key names
        self.ca = self.channel_attention
        self.sa = self.spatial_attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Channel Attention (What to look for)
        x_out = x * self.channel_attention(x)
        # Step 2: Spatial Attention (Where to look)
        x_out = x_out * self.spatial_attention(x_out)
        return x_out
