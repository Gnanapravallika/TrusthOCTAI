"""Multi-scale feature fusion module for TrustOCT framework."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFusion(nn.Module):
    """Fuses multi-scale features from Layer 3 and Layer 4 of ResNet."""

    def __init__(self, layer3_channels: int = 1024, layer4_channels: int = 2048, out_channels: int = 512):
        """Initialize multi-scale feature fusion.

        Args:
            layer3_channels: Number of channels in intermediate features (Layer 3).
            layer4_channels: Number of channels in deep features (Layer 4).
            out_channels: Output channels after channel projection.
        """
        super().__init__()
        # Concat channels: 1024 + 2048 = 3072
        in_channels = layer3_channels + layer4_channels
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, layer3_out: torch.Tensor, layer4_out: torch.Tensor) -> torch.Tensor:
        """Forward pass fusing features.

        Args:
            layer3_out: Intermediate feature maps of shape [B, layer3_channels, H3, W3].
            layer4_out: Deep feature maps of shape [B, layer4_channels, H4, W4].

        Returns:
            Fused feature map of shape [B, out_channels, H4, W4].
        """
        # Downsample Layer 3 to match Layer 4 resolution (typically 14x14 downsampled to 7x7)
        h4, w4 = layer4_out.shape[2], layer4_out.shape[3]
        layer3_resized = F.interpolate(layer3_out, size=(h4, w4), mode="bilinear", align_corners=False)

        # Concatenate features along the channel dimension
        fused = torch.cat([layer3_resized, layer4_out], dim=1)
        
        # Project to target dimension
        out = self.proj(fused)
        out = self.bn(out)
        return self.relu(out)
