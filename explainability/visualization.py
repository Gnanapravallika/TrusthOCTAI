"""Visualization engine for Grad-CAM and LayerCAM attribution overlays on OCT scans."""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from explainability.layercam import LayerCAM


def compare_and_save_visualizations(
    model: torch.nn.Module,
    target_layers_gradcam: list,
    target_layers_layercam: list,
    image_path: str,
    target_class: int = None,
    output_dir: str = "outputs/msf_cbam_resnet50/layercam",
    prefix: str = "scan"
):
    """Generate and save side-by-side Original, Grad-CAM, and LayerCAM visual overlays."""
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device

    # Read original image
    raw_img = Image.open(image_path).convert('RGB').resize((224, 224))
    rgb_img = np.float32(raw_img) / 255.0

    # Normalize tensor for model input
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm_img = (rgb_img - mean) / std
    input_tensor = torch.tensor(norm_img).permute(2, 0, 1).unsqueeze(0).to(device)

    # 1. Grad-CAM
    try:
        gradcam = GradCAM(model=model, target_layers=target_layers_gradcam)
        grayscale_gradcam = gradcam(input_tensor=input_tensor, targets=None)[0]
        gradcam_overlay = show_cam_on_image(rgb_img, grayscale_gradcam, use_rgb=True)
    except Exception as e:
        gradcam_overlay = (rgb_img * 255).astype(np.uint8)

    # 2. LayerCAM
    try:
        layercam = LayerCAM(model=model, target_layers=target_layers_layercam)
        grayscale_layercam = layercam.generate(input_tensor, target_class=target_class)
        layercam.remove_hooks()
        layercam_overlay = show_cam_on_image(rgb_img, grayscale_layercam, use_rgb=True)
    except Exception as e:
        layercam_overlay = (rgb_img * 255).astype(np.uint8)

    # Save individual images
    cv2.imwrite(os.path.join(output_dir, f"{prefix}_original.png"), cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, f"{prefix}_gradcam.png"), cv2.cvtColor(gradcam_overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, f"{prefix}_layercam.png"), cv2.cvtColor(layercam_overlay, cv2.COLOR_RGB2BGR))

    print(f"✅ Attribution maps saved for {prefix} at {output_dir}")
