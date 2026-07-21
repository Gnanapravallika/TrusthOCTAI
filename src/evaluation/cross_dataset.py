"""Cross-dataset external validation pipeline for TrustOCT framework."""

import os
import sys
from typing import Dict, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from src.evaluation.classification import calculate_classification_metrics
    from src.evaluation.calibration import generate_calibration_stats
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.evaluation.classification import calculate_classification_metrics
    from src.evaluation.calibration import generate_calibration_stats


def evaluate_cross_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    head_type: str = "edl",
    num_classes: int = 4
) -> Dict[str, Union[float, list]]:
    """Run evaluation of trained model on an external target dataset.

    Args:
        model: Trained TrustOCT model.
        loader: Target dataset DataLoader.
        device: PyTorch compute device.
        head_type: Configured head type ('softmax', 'edl').
        num_classes: Number of classification targets.

    Returns:
        Dict containing validation classification and calibration metrics.
    """
    model.eval()
    
    all_targets = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            
            outputs = model(images)
            
            if head_type.lower() == "edl":
                # evidential head returns (evidence, alpha)
                _, alpha = outputs
                # Dirichlet strength S = sum(alpha)
                S = torch.sum(alpha, dim=1, keepdim=True)
                probs = alpha / S
                preds = torch.argmax(alpha, dim=1)
            else:
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    
    # Calculate classification metrics
    clf_metrics = calculate_classification_metrics(y_true, y_pred, y_prob, num_classes=num_classes)
    
    # Calculate calibration metrics
    cal_metrics = generate_calibration_stats(y_true, y_prob, num_classes=num_classes)
    
    # Combine results
    results = {**clf_metrics, **cal_metrics}
    return results
