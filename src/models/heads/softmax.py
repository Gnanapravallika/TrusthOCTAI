"""Standard softmax classification head for TrustOCT framework."""

import os
import sys
import torch
import torch.nn as nn

try:
    from src.models.heads.base_head import BaseHead
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
    from src.models.heads.base_head import BaseHead


class SoftmaxHead(BaseHead):
    """Standard classifier head outputting class logits."""

    def __init__(self, in_features: int, num_classes: int, dropout_prob: float = 0.5):
        """Initialize standard classification head.

        Args:
            in_features: Input features dimensions.
            num_classes: Number of classification targets.
            dropout_prob: Dropout probability.
        """
        super().__init__(in_features, num_classes)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input feature vector of shape [Batch, in_features].

        Returns:
            Output logits of shape [Batch, num_classes].
        """
        x = self.dropout(x)
        return self.fc(x)
