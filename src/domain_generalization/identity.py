"""Identity domain generalization layer for TrustOCT framework."""

import torch
import torch.nn as nn


class IdentityDG(nn.Module):
    """Identity Domain Generalization module (performs no statistics perturbation)."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W] or [B, D].

        Returns:
            Unmodified input tensor.
        """
        return x
