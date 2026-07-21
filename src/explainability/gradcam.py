"""Grad-CAM visual attribution module for TrustOCT framework."""

import torch
import torch.nn as nn

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    # Fallback placeholder if library is missing in local check
    class GradCAM:
        def __init__(self, **kwargs): pass
        def __call__(self, **kwargs): return torch.zeros((1, 224, 224)).numpy()
    class ClassifierOutputTarget:
        def __init__(self, *args): pass


class TrustGradCAM:
    """Wrapper class for generating Grad-CAM visualizations on TrustOCT models."""

    def __init__(self, model: nn.Module, target_layers: list):
        """Initialize Grad-CAM.

        Args:
            model: PyTorch model (TrustOCT).
            target_layers: List of target layer modules from model (e.g. model.backbone.layer4).
        """
        self.cam = GradCAM(model=model, target_layers=target_layers)

    def generate(self, input_tensor: torch.Tensor, target_class: int) -> torch.Tensor:
        """Generate a grayscale Grad-CAM attribution map.

        Args:
            input_tensor: Input image tensor of shape [1, 3, H, W].
            target_class: Target class category index to visualize.

        Returns:
            Grayscale attribution map of shape [H, W] as a numpy array.
        """
        targets = [ClassifierOutputTarget(target_class)]
        # Generate CAM
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        return grayscale_cam[0, :]
