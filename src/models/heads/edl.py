"""Evidential Dirichlet output head for TrustOCT framework."""

import os
import sys
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from src.models.heads.base_head import BaseHead
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
    from src.models.heads.base_head import BaseHead


class EvidentialHead(BaseHead):
    """Evidential classification head outputting positive evidence parameters."""

    def __init__(self, in_features: int, num_classes: int, dropout_prob: float = 0.5):
        """Initialize evidential head.

        Args:
            in_features: Input features dimensions.
            num_classes: Number of target classes.
            dropout_prob: Dropout probability.
        """
        super().__init__(in_features, num_classes)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input feature vector of shape [Batch, in_features].

        Returns:
            Tuple of:
                - evidence: Positive evidence parameters of shape [Batch, num_classes] (>= 0).
                - alpha: Dirichlet concentration parameters ($\alpha = \text{evidence} + 1$).
        """
        x = self.dropout(x)
        logits = self.fc(x)
        
        # Softplus ensures positive evidence values (>= 0)
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        
        return evidence, alpha
