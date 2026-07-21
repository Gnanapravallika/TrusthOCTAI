"""Convolutional Block Attention Module (CBAM) for TrustOCT framework."""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel Attention Module of CBAM."""

    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
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
    """Spatial Attention Module of CBAM."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(concat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """Unified CBAM Attention Block."""

    def __init__(self, gate_channels: int, reduction_ratio: int = 16, spatial_kernel_size: int = 7):
        """Initialize CBAM module.

        Args:
            gate_channels: Input channel size.
            reduction_ratio: Reduction ratio for channel attention MLP.
            spatial_kernel_size: Convolution kernel size for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(gate_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Attention-weighted tensor of same shape.
        """
        # Apply Channel Attention
        x = x * self.channel_attention(x)
        # Apply Spatial Attention
        x = x * self.spatial_attention(x)
        return x
