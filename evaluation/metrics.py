"""Diagnostic performance metrics and confusion matrix plotting for TrustOCT."""

import os
from typing import Dict, List, Union
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix
)


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int = 4
) -> Dict[str, Union[float, list]]:
    """Compute standard metrics including macro accuracy, F1, MCC, and Specificity."""
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Specificity
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

    try:
        auc_score = roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="macro", labels=list(range(num_classes))
        )
    except Exception:
        auc_score = 0.5

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": mean_specificity,
        "f1_score": float(f1),
        "mcc": float(mcc),
        "kappa": float(kappa),
        "roc_auc": float(auc_score),
        "confusion_matrix": cm.tolist()
    }


def plot_confusion_matrix(
    cm: list,
    classes: List[str],
    save_path: str = "confusion_matrix.png",
    normalize: bool = False
) -> None:
    """Generate and save a confusion matrix heatmap."""
    matrix = np.array(cm)
    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=(8, 8), dpi=300)
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = format(matrix[i, j], fmt)
            plt.text(
                j, i, val,
                horizontalalignment="center",
                color="white" if matrix[i, j] > thresh else "black",
                fontsize=12,
                fontweight="bold"
            )

    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()

    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")
