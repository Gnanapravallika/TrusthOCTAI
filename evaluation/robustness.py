"""Robustness evaluation, OOD detection, and selective prediction routing checks for TrustOCT."""

from typing import Dict
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def calculate_ood_detection_metrics(
    id_uncertainties: np.ndarray,
    ood_uncertainties: np.ndarray
) -> Dict[str, float]:
    """Calculate OOD detection metrics using AUROC, AUPR, and FPR95."""
    n_id = len(id_uncertainties)
    n_ood = len(ood_uncertainties)
    
    if n_id == 0 or n_ood == 0:
        return {"auroc": 0.5, "aupr_ood": 0.5, "fpr95": 1.0}

    y_true = np.concatenate([np.zeros(n_id), np.ones(n_ood)])
    y_scores = np.concatenate([id_uncertainties, ood_uncertainties])

    auroc = roc_auc_score(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    aupr_ood = auc(recall, precision)

    threshold = np.percentile(ood_uncertainties, 5)
    fpr95 = np.mean(id_uncertainties >= threshold)

    return {
        "auroc": float(auroc),
        "aupr_ood": float(aupr_ood),
        "fpr95": float(fpr95)
    }


def evaluate_selective_prediction(
    uncertainties: np.ndarray,
    accuracies: np.ndarray,
    thresholds: np.ndarray
) -> Dict[str, list]:
    """Calculate coverage vs risk for selective prediction routing."""
    coverages = []
    risks = []
    
    for t in thresholds:
        accepted = (uncertainties < t)
        if np.sum(accepted) > 0:
            coverage = np.mean(accepted)
            risk = 1.0 - np.mean(accuracies[accepted])
        else:
            coverage = 0.0
            risk = 0.0
        coverages.append(float(coverage))
        risks.append(float(risk))
        
    return {
        "coverages": coverages,
        "risks": risks
    }
