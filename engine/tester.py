"""Tester loop module for evaluating TrustOCT models."""

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    head_type: str = "edl"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model over test_loader and collect predictions, probabilities, and labels."""
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)

            if head_type.lower() == "edl":
                _, alpha = outputs
                probs = alpha / torch.sum(alpha, dim=1, keepdim=True)
                preds = torch.argmax(alpha, dim=1)
            else:
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)

            all_targets.append(targets.numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    return y_true, y_pred, y_prob
