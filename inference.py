"""Inference script for running predictions and uncertainty estimation on single OCT scans."""

import argparse
import os
import sys
import numpy as np
import torch
import yaml
from PIL import Image

# Add project root to python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from models.trustoct import build_model
from datasets.transforms import get_val_transforms
from datasets.dataset import CLASS_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description="TrustOCT Inference Script")
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to input OCT scan."
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model.yaml",
        help="Path to model config YAML."
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="configs/train.yaml",
        help="Path to train config YAML."
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        required=True,
        help="Path to trained model weights."
    )
    return parser.parse_args()


def run_inference():
    args = parse_args()

    # Load configs
    with open(args.model_config, "r") as f:
        model_cfg = yaml.safe_load(f)
    with open(args.train_config, "r") as f:
        train_cfg = yaml.safe_load(f)

    # Build model and load weights
    model = build_model(args.model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(args.weights_path, map_location=device)
    # Check if checkpoint is dict with model_state or just model state
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()

    # Prep transforms
    transforms = get_val_transforms({**model_cfg, **train_cfg})

    # Load and transform image
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        return

    image = Image.open(args.image_path).convert("RGB")
    image_np = np.array(image)
    augmented = transforms(image=image_np)
    img_tensor = augmented["image"].unsqueeze(0).to(device)

    # Run inference
    head_type = model_cfg.get("head", "edl").lower()
    num_classes = model_cfg.get("num_classes", 4)

    with torch.no_grad():
        outputs = model(img_tensor)
        if head_type == "edl":
            evidence, alpha = outputs
            S = torch.sum(alpha, dim=1)
            probs = alpha / S
            uncertainty = num_classes / S.item()
            pred_idx = torch.argmax(probs, dim=1).item()
            conf = probs[0, pred_idx].item()
        else:
            probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            conf = probs[0, pred_idx].item()
            uncertainty = 0.0

    print("=" * 50)
    print(f"OCT Scan: {args.image_path}")
    print(f"Prediction: {CLASS_NAMES[pred_idx]} ({conf * 100:.2f}% confidence)")
    if head_type == "edl":
        print(f"Epistemic Uncertainty: {uncertainty:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    run_inference()
