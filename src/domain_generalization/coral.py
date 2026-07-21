"""CORAL domain covariance alignment module for TrustOCT framework."""

import torch
import torch.nn as nn


def compute_covariance(x: torch.Tensor) -> torch.Tensor:
    """Compute the covariance matrix of input features.

    Args:
        x: Input feature tensor of shape [B, D].

    Returns:
        Covariance matrix of shape [D, D].
    """
    n = x.size(0)
    if n <= 1:
        return torch.zeros((x.size(1), x.size(1)), device=x.device)

    # Subtract mean
    mean = torch.mean(x, dim=0, keepdim=True)
    x_centered = x - mean
    
    # Compute covariance
    covariance = torch.matmul(x_centered.t(), x_centered) / (n - 1)
    return covariance


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Deep CORAL loss (Sun & Saenko, ECCV 2016).

    Args:
        source: Source domain features of shape [B_s, D].
        target: Target domain features of shape [B_t, D].

    Returns:
        Scalar CORAL loss tensor.
    """
    d = source.size(1)
    
    source_cov = compute_covariance(source)
    target_cov = compute_covariance(target)
    
    # Frobenius norm squared
    loss = torch.mean((source_cov - target_cov) ** 2) / (4 * d * d)
    return loss


class CoralFeatureAlignment(nn.Module):
    """Wrapper module that passes features through and stores them for CORAL loss computation."""

    def __init__(self):
        super().__init__()
        self.last_features = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Stores features in self.last_features.

        Args:
            x: Input feature maps or vectors of shape [B, C, H, W] or [B, D].

        Returns:
            Unmodified input features.
        """
        # If spatial features, apply Global Average Pooling first
        if x.dim() == 4:
            self.last_features = torch.mean(x, dim=[2, 3])
        else:
            self.last_features = x
            
        return x
