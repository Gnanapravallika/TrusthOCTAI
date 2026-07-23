"""ResNet50 backbone extracting intermediate layer 3 and layer 4 features."""

from typing import Tuple
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class ResNet50Backbone(nn.Module):
    """ResNet50 backbone extracting intermediate layer 3 and layer 4 features."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        
        layer3_out = self.layer3(x)
        layer4_out = self.layer4(layer3_out)

        return layer3_out, layer4_out
