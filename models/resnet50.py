"""ResNet-50 dual-layer feature extractor outputting Layer 3 and Layer 4 representations."""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Backbone(nn.Module):
    """ResNet-50 backbone extracting multi-scale feature maps from Layer 3 and Layer 4."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base_resnet = resnet50(weights=weights)

        # Initial stem layers
        self.conv1 = base_resnet.conv1
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool

        # ResNet bottleneck layers
        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3  # Output: [B, 1024, 14, 14]
        self.layer4 = base_resnet.layer4  # Output: [B, 2048, 7, 7]

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        layer3_out = self.layer3(x)  # [B, 1024, 14, 14]
        layer4_out = self.layer4(layer3_out)  # [B, 2048, 7, 7]

        return layer3_out, layer4_out
