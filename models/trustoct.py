"""TrustOCT Framework Unified Architecture.

Combines ResNet-50 backbone, Multi-Scale Feature Fusion (MSF), Dual CBAM Attention,
and Evidential / Softmax Decision Head.
"""

import os
import random
from typing import Tuple, Dict, Union
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet50 import ResNet50Backbone
from models.msf import MultiScaleFusion
from models.cbam import CBAM


# =====================================================================
# Prediction Heads
# =====================================================================

class SoftmaxHead(nn.Module):
    """Standard classification head with Dropout and FC projection."""

    def __init__(self, in_features: int = 2048, num_classes: int = 4, dropout_prob: float = 0.5):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.fc(x)


class EvidentialHead(nn.Module):
    """Evidential classification head producing Dirichlet evidence parameters."""

    def __init__(self, in_features: int = 2048, num_classes: int = 4, dropout_prob: float = 0.5):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.dropout(x)
        logits = self.fc(x)
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        return evidence, alpha


# =====================================================================
# Domain Generalization Components
# =====================================================================

class MixStyle(nn.Module):
    """MixStyle module (Zhou et al., ICLR 2021) for domain generalization."""

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

        perm = torch.randperm(batch_size).to(x.device)
        mu_shuffled = mu[perm]
        sig_shuffled = sig[perm]

        beta_dist = torch.distributions.Beta(self.alpha, self.alpha)
        lmda = beta_dist.sample((batch_size, 1, 1, 1)).to(x.device)

        mu_mixed = lmda * mu + (1.0 - lmda) * mu_shuffled
        sig_mixed = lmda * sig + (1.0 - lmda) * sig_shuffled

        return x_norm * sig_mixed + mu_mixed


# =====================================================================
# TrustOCT Unified Model
# =====================================================================

class TrustOCT(nn.Module):
    """
    TrustOCT Unified Framework Model.
    Architecture:
      Input (3, 224, 224)
      -> ResNet50 Backbone (Layer3: [B, 1024, 14, 14], Layer4: [B, 2048, 7, 7])
      -> MSF Fusion ([B, 2048, 7, 7])
      -> Dual CBAM Attention ([B, 2048, 7, 7])
      -> AdaptiveAvgPool2d((1, 1)) -> Linear(2048, 4)
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        use_multiscale: bool = True,
        use_cbam: bool = True,
        dg_type: str = "mixstyle",
        head_type: str = "softmax",
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

        # 1. ResNet-50 Dual-Layer Feature Extractor
        self.backbone = ResNet50Backbone(pretrained=pretrained)
        self.feature_channels = 2048

        # 2. Domain Generalization
        if self.dg_type == "mixstyle":
            self.dg = MixStyle(p=dg_p, alpha=dg_alpha)
        else:
            self.dg = nn.Identity()

        # 3. Multi-Scale Feature Fusion (MSF)
        if self.use_multiscale:
            self.fusion = MultiScaleFusion(in_channels_l3=1024, out_channels=2048)
        else:
            self.fusion = None

        # 4. Dual CBAM Attention Module
        if self.use_cbam:
            self.attention = CBAM(gate_channels=2048, reduction_ratio=16, spatial_kernel_size=7)
        else:
            self.attention = None

        # 5. Global Average Pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 6. Classifier Head
        if self.head_type == "softmax":
            self.head = SoftmaxHead(in_features=2048, num_classes=num_classes, dropout_prob=dropout_prob)
        elif self.head_type == "edl":
            self.head = EvidentialHead(in_features=2048, num_classes=num_classes, dropout_prob=dropout_prob)
        else:
            raise ValueError(f"Unknown head type '{head_type}'")

    def forward(self, x: torch.Tensor, return_features: bool = False):
        # Extract dual-layer feature maps
        layer3_out, layer4_out = self.backbone(x)

        # Domain generalization on Layer 3
        if self.dg_type == "mixstyle":
            layer3_out = self.dg(layer3_out)

        # Multi-scale feature fusion
        if self.use_multiscale and self.fusion is not None:
            x_fused = self.fusion(layer3_out, layer4_out)
        else:
            x_fused = layer4_out

        # Dual CBAM attention refinement
        if self.use_cbam and self.attention is not None:
            x_att = self.attention(x_fused)
        else:
            x_att = x_fused

        # Global average pooling & flattening
        feat = self.pool(x_att)
        feat = torch.flatten(feat, start_dim=1)

        # Classification output
        out = self.head(feat)

        if return_features:
            return {
                "output": out,
                "embedding": feat,
                "attention_map": x_att,
                "fused_map": x_fused
            }
        return out


def build_model(model_config_path: str) -> nn.Module:
    """Build TrustOCT model instance dynamically from YAML config."""
    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"Config path missing: {model_config_path}")

    with open(model_config_path, "r") as f:
        config = yaml.safe_load(f)

    backbone_name = config.get("backbone", "resnet50")
    pretrained = config.get("pretrained", True)
    use_multiscale = (config.get("feature_module", "multiscale") == "multiscale")
    use_cbam = (config.get("attention", "cbam") == "cbam")
    dg_type = config.get("domain_generalization", "mixstyle")
    head_type = config.get("head", "softmax")
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
        dropout_prob=dropout_prob
    )

    print(f"Successfully compiled TrustOCT Model Architecture:")
    print(f"  • Backbone      : {backbone_name} (Pretrained: {pretrained})")
    print(f"  • Feature Fusion: {'MSF (1024 -> 2048 Stride 2 Add)' if use_multiscale else 'None'}")
    print(f"  • Attention     : {'Dual CBAM (Channel + Spatial)' if use_cbam else 'None'}")
    print(f"  • Head Type     : {head_type}")
    print(f"  • Output Classes: {num_classes}")

    return model
