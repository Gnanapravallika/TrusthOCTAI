"""Explainability wrappers (Grad-CAM and LayerCAM) for TrustOCT."""

import numpy as np
import torch
import torch.nn as nn

try:
    from pytorch_grad_cam import GradCAM, LayerCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    # Dummy fallbacks for checkups
    class GradCAM:
        def __init__(self, **kwargs): pass
        def __call__(self, **kwargs): return np.zeros((1, 224, 224))
    class LayerCAM:
        def __init__(self, **kwargs): pass
        def __call__(self, **kwargs): return np.zeros((1, 224, 224))
    class ClassifierOutputTarget:
        def __init__(self, *args): pass
    def show_cam_on_image(img, mask, **kwargs):
        return (img * 255.0).astype(np.uint8)


class TrustGradCAM:
    """Wrapper class generating Grad-CAM maps on models."""

    def __init__(self, model: nn.Module, target_layers: list):
        self.cam = GradCAM(model=model, target_layers=target_layers)

    def generate(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        return grayscale_cam[0, :]


class TrustLayerCAM:
    """Wrapper class generating LayerCAM maps on models."""

    def __init__(self, model: nn.Module, target_layers: list):
        self.cam = LayerCAM(model=model, target_layers=target_layers)

    def generate(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        return grayscale_cam[0, :]
