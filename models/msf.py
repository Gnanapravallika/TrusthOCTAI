"""Multi-Scale Feature Fusion (MSF) Module.

Fuses Layer 3 fine spatial features [B, 1024, 14, 14] with Layer 4 semantic features [B, 2048, 7, 7]
via 1x1 stride=2 projection followed by element-wise addition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFusion(nn.Module):
    """Multi-Scale Feature Fusion combining fine-grained and global semantic representations."""

    def __init__(self, in_channels_l3: int = 1024, out_channels: int = 2048):
        super().__init__()
        # 1x1 Conv with stride 2 to project [B, 1024, 14, 14] -> [B, 2048, 7, 7]
        self.proj = nn.Conv2d(
            in_channels=in_channels_l3,
            out_channels=out_channels,
            kernel_size=1,
            stride=2,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Alias container for checkpoint key matching if needed
        self.fusion_conv = nn.Sequential(
            self.proj,
            self.bn,
            self.relu
        )

    def forward(self, layer3_out: torch.Tensor, layer4_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            layer3_out: [B, 1024, 14, 14]
            layer4_out: [B, 2048, 7, 7]
        Returns:
            F_fused: [B, 2048, 7, 7]
        """
        # If layer3 spatial dimensions match layer4, stride 1; if 14x14 vs 7x7, use 1x1 stride=2
        h3, w3 = layer3_out.shape[2], layer3_out.shape[3]
        h4, w4 = layer4_out.shape[2], layer4_out.shape[3]

        if h3 != h4 or w3 != w4:
            l3_proj = self.fusion_conv(layer3_out)
        else:
            l3_proj = F.interpolate(layer3_out, size=(h4, w4), mode='bilinear', align_corners=False)
            l3_proj = self.fusion_conv[0](l3_proj)
            l3_proj = self.fusion_conv[1](l3_proj)
            l3_proj = self.fusion_conv[2](l3_proj)

        fused = l3_proj + layer4_out
        return fused
