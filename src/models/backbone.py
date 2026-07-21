"""Backbone loaders for standard CNN and transformer models in TrustOCT."""

from typing import Tuple
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class ResNet50Backbone(nn.Module):
    """Wrapper around torchvision ResNet50 to extract multiscale features."""

    def __init__(self, pretrained: bool = True):
        """Initialize ResNet50 backbone.

        Args:
            pretrained: Load pretrained ImageNet weights.
        """
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        # Base stem
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # Residual layers
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass extracting intermediate features.

        Args:
            x: Input image tensor of shape [Batch, 3, H, W].

        Returns:
            Tuple of:
                - layer3_out: Feature maps of shape [Batch, 1024, H/16, W/16]
                - layer4_out: Feature maps of shape [Batch, 2048, H/32, W/32]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        
        layer3_out = self.layer3(x)
        layer4_out = self.layer4(layer3_out)

        return layer3_out, layer4_out


def build_backbone(name: str, pretrained: bool = True) -> nn.Module:
    """Factory function to build backbones.

    Args:
        name: Name of the backbone ('resnet50').
        pretrained: Load pretrained ImageNet weights.

    Returns:
        nn.Module backbone instance.
    """
    if name.lower() == "resnet50":
        return ResNet50Backbone(pretrained=pretrained)
    else:
        raise NotImplementedError(f"Backbone '{name}' is not yet implemented.")
