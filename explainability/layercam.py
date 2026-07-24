"""LayerCAM implementation for fine-grained multi-layer attribution in CNNs."""

import torch
import torch.nn.functional as F
import numpy as np
import cv2


class LayerCAM:
    """LayerCAM (Jiang et al., IEEE T-MM 2021) for fine-grained layer attribution."""

    def __init__(self, model: torch.nn.Module, target_layers: list):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []
        self.activations = []
        self.handles = []

        self._register_hooks()

    def _register_hooks(self):
        for layer in self.target_layers:
            self.handles.append(layer.register_forward_hook(self._forward_hook))
            self.handles.append(layer.register_full_backward_hook(self._backward_hook))

    def _forward_hook(self, module, input, output):
        self.activations.append(output)

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients.insert(0, grad_output[0])

    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        self.model.eval()
        self.gradients = []
        self.activations = []

        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()

        cam_list = []
        for act, grad in zip(self.activations, self.gradients):
            act = act[0]   # [C, H, W]
            grad = grad[0] # [C, H, W]

            # Positive gradients weight element-wise activations
            weights = F.relu(grad)
            cam = torch.sum(weights * act, dim=0)
            cam = F.relu(cam)

            cam = cam.detach().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam_list.append(cam)

        # Average multi-layer CAMs if multiple target layers passed
        final_cam = np.mean(cam_list, axis=0)
        final_cam = cv2.resize(final_cam, (input_tensor.shape[3], input_tensor.shape[2]))
        return final_cam

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
