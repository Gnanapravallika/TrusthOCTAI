"""Classification metrics evaluation for TrustOCT framework."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix
)
import torch
from typing import Dict, Union, Tuple


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int = 4
) -> Dict[str, Union[float, np.ndarray]]:
    """Compute comprehensive diagnostic metrics.

    Args:
        y_true: Ground truth integer array of shape [N].
        y_pred: Predicted class indices array of shape [N].
        y_prob: Prediction probability distribution array of shape [N, num_classes].
        num_classes: Number of disease targets.

    Returns:
        Dict containing standard metric values.
    """
    # 1. Basic metrics
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # 2. Precision, Recall, Specificity, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Per-class specificity calculation from confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificities = []
    for i in range(num_classes):
        temp_cm = np.delete(cm, i, axis=0)
        temp_cm = np.delete(temp_cm, i, axis=1)
        tn = temp_cm.sum()
        fp = cm[:, i].sum() - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(specificity)
    mean_specificity = float(np.mean(specificities))

    # 3. ROC-AUC calculation
    try:
        # Multiclass ROC-AUC (OVR mode)
        auc = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro",
            labels=list(range(num_classes))
        )
    except Exception:
        auc = 0.5  # Fallback for unrepresented classes in test splits

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": mean_specificity,
        "f1_score": float(f1),
        "mcc": float(mcc),
        "kappa": float(kappa),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist()
    }
