"""Evidential heads, style-generalization layers, and TrustOCT model builder."""

import os
import random
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Union
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet50 import ResNet50Backbone
from models.msf import MultiScaleFusion
from models.cbam import CBAM


# =====================================================================
# 1. Prediction Heads
# =====================================================================

class BaseHead(nn.Module, ABC):
    """Abstract base class for all prediction heads."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor):
        pass


class SoftmaxHead(BaseHead):
    """Standard softmax classifier head producing class logits."""

    def __init__(self, in_features: int, num_classes: int, dropout_prob: float = 0.5):
        super().__init__(in_features, num_classes)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.fc(x)


class EvidentialHead(BaseHead):
    """Evidential classification head outputting positive Dirichlet evidence parameters."""

    def __init__(self, in_features: int, num_classes: int, dropout_prob: float = 0.5):
        super().__init__(in_features, num_classes)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.dropout(x)
        logits = self.fc(x)
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        return evidence, alpha


# =====================================================================
# 2. Domain Generalization Components
# =====================================================================

class MixStyle(nn.Module):
    """MixStyle module (Zhou et al., ICLR 2021) for style statistics blending."""

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or random.random() > self.p:
            return x

        batch_size = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()

        x_norm = (x - mu) / sig

        # Shuffle styles
        perm = torch.randperm(batch_size).to(x.device)
        mu_shuffled = mu[perm]
        sig_shuffled = sig[perm]

        beta_dist = torch.distributions.Beta(self.alpha, self.alpha)
        lmda = beta_dist.sample((batch_size, 1, 1, 1)).to(x.device)

        mu_mixed = lmda * mu + (1.0 - lmda) * mu_shuffled
        sig_mixed = lmda * sig + (1.0 - lmda) * sig_shuffled

        return x_norm * sig_mixed + mu_mixed


def compute_covariance(x: torch.Tensor) -> torch.Tensor:
    """Compute the covariance matrix of input features."""
    n = x.size(0)
    if n <= 1:
        return torch.zeros((x.size(1), x.size(1)), device=x.device)
    mean = torch.mean(x, dim=0, keepdim=True)
    x_centered = x - mean
    covariance = torch.matmul(x_centered.t(), x_centered) / (n - 1)
    return covariance


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Deep CORAL loss."""
    d = source.size(1)
    source_cov = compute_covariance(source)
    target_cov = compute_covariance(target)
    loss = torch.mean((source_cov - target_cov) ** 2) / (4 * d * d)
    return loss


class CoralFeatureAlignment(nn.Module):
    """Wrapper storing features for CORAL covariance alignment."""

    def __init__(self):
        super().__init__()
        self.last_features = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            self.last_features = torch.mean(x, dim=[2, 3])
        else:
            self.last_features = x
        return x


# =====================================================================
# 3. TrustOCT Assembler
# =====================================================================

class TrustOCT(nn.Module):
    """Unified framework architecture combining swappable component layers."""

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        use_multiscale: bool = True,
        use_cbam: bool = True,
        dg_type: str = "mixstyle",
        head_type: str = "edl",
        num_classes: int = 4,
        dropout_prob: float = 0.5,
        dg_p: float = 0.5,
        dg_alpha: float = 0.1
    ):
        super().__init__()
        self.use_multiscale = use_multiscale
        self.use_cbam = use_cbam
        self.dg_type = dg_type.lower()
        self.head_type = head_type.lower()

        # 1. Backbone wrapper
        if backbone_name.lower() == "resnet50":
            self.backbone = ResNet50Backbone(pretrained=pretrained)
            self.layer3_channels = 1024
            self.layer4_channels = 2048
        else:
            raise NotImplementedError(f"Backbone '{backbone_name}' not implemented.")

        # 2. Domain Generalization
        if self.dg_type == "mixstyle":
            self.dg = MixStyle(p=dg_p, alpha=dg_alpha)
        elif self.dg_type == "coral":
            self.dg = CoralFeatureAlignment()
        else:
            self.dg = nn.Identity()

        # 3. Multi-scale fusion
        if self.use_multiscale:
            self.fusion = MultiScaleFusion(
                layer3_channels=self.layer3_channels,
                layer4_channels=self.layer4_channels,
                out_channels=512
            )
            self.feature_channels = 512
        else:
            self.feature_channels = self.layer4_channels

        # 4. Attention
        if self.use_cbam:
            self.attention = CBAM(gate_channels=self.feature_channels)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 5. Classifier Heads
        if self.head_type == "softmax":
            self.head = SoftmaxHead(self.feature_channels, num_classes, dropout_prob)
        elif self.head_type == "edl":
            self.head = EvidentialHead(self.feature_channels, num_classes, dropout_prob)
        else:
            raise ValueError(f"Unknown head type '{head_type}'")

        # 6. Weight Initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        layer3_out, layer4_out = self.backbone(x)

        if self.dg_type == "mixstyle":
            layer3_out = self.dg(layer3_out)

        if self.use_multiscale:
            x_fused = self.fusion(layer3_out, layer4_out)
        else:
            x_fused = layer4_out

        if self.use_cbam:
            x_att = self.attention(x_fused)
        else:
            x_att = x_fused

        if self.dg_type == "coral":
            x_att = self.dg(x_att)

        feat = self.pool(x_att)
        feat = torch.flatten(feat, start_dim=1)
        
        logits_out = self.head(feat)

        if return_features:
            return {
                "logits": logits_out,
                "embedding": feat,
                "attention": x_att,
                "fusion": x_fused
            }
        return logits_out


# =====================================================================
# 4. Model Builder
# =====================================================================

def build_model(model_config_path: str) -> nn.Module:
    """Build the TrustOCT model dynamically from config file."""
    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"Model config not found at {model_config_path}")
        
    with open(model_config_path, "r") as f:
        config = yaml.safe_load(f)

    backbone_name = config.get("backbone", "resnet50")
    pretrained = config.get("pretrained", True)
    use_multiscale = (config.get("feature_module", "multiscale") == "multiscale")
    use_cbam = (config.get("attention", "cbam") == "cbam")
    
    dg_type = config.get("domain_generalization", "mixstyle")
    mixstyle_cfg = config.get("mixstyle", {})
    dg_p = mixstyle_cfg.get("mix_prob", 0.5)
    dg_alpha = mixstyle_cfg.get("alpha", 0.1)

    head_type = config.get("head", "edl")
    num_classes = config.get("num_classes", 4)
    dropout_prob = config.get("dropout", 0.5)

    model = TrustOCT(
        backbone_name=backbone_name,
        pretrained=pretrained,
        use_multiscale=use_multiscale,
        use_cbam=use_cbam,
        dg_type=dg_type,
        head_type=head_type,
        num_classes=num_classes,
        dropout_prob=dropout_prob,
        dg_p=dg_p,
        dg_alpha=dg_alpha
    )

    print(f"Successfully compiled TrustOCT Model V1.0:")
    print(f"  • Backbone:     {backbone_name} (Pretrained: {pretrained})")
    print(f"  • Feature Fusion: {config.get('feature_module', 'multiscale')}")
    print(f"  • Attention:     {config.get('attention', 'cbam')}")
    print(f"  • Generalization: {dg_type}")
    print(f"  • Decision Head:  {head_type}")
    print(f"  • Classes:       {num_classes}")

    return model
