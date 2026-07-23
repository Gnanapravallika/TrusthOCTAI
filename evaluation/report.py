"""Cross-dataset loops and final evaluation reports generation for TrustOCT."""

import json
import os
from typing import Dict, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.metrics import calculate_classification_metrics
from evaluation.calibration import calculate_ece, calculate_brier_score


def evaluate_cross_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    head_type: str = "edl",
    num_classes: int = 4
) -> Dict[str, Union[float, list]]:
    """Evaluate trained model on an external dataloader."""
    model.eval()
    
    all_targets = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            
            if head_type.lower() == "edl":
                _, alpha = outputs
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
    
    clf_metrics = calculate_classification_metrics(y_true, y_pred, y_prob, num_classes=num_classes)
    ece = calculate_ece(y_true, y_prob)
    brier = calculate_brier_score(y_true, y_prob, num_classes=num_classes)
    
    results = {
        **clf_metrics,
        "ece": ece,
        "brier_score": brier
    }
    return results


def save_reports(
    results: dict,
    output_dir: str,
    experiment_name: str
) -> None:
    """Save results as JSON and print reports."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "evaluation_report.json")
    
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"=== TrustOCT Evaluation Report: {experiment_name} ===")
    print(f"Accuracy:    {results.get('accuracy', 0.0):.4f}")
    print(f"F1-score:    {results.get('f1_score', 0.0):.4f}")
    print(f"Kappa:       {results.get('kappa', 0.0):.4f}")
    print(f"ECE:         {results.get('ece', 0.0):.4f}")
    print(f"Brier Score: {results.get('brier_score', 0.0):.4f}")
    print(f"Report saved to: {report_path}")
