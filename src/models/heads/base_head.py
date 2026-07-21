"""Base abstract class for TrustOCT prediction heads."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseHead(nn.Module, ABC):
    """Abstract base class for all classification and uncertainty heads."""

    def __init__(self, in_features: int, num_classes: int):
        """Initialize head.

        Args:
            in_features: Channel size of input features.
            num_classes: Target number of output classes.
        """
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input feature tensor of shape [Batch, in_features].

        Returns:
            Output prediction tensor.
        """
        pass
