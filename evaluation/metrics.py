"""Comprehensive medical classification performance evaluation metrics."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix
)


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    """Calculate accuracy, macro precision, recall, F1-score, MCC, and ROC-AUC."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    roc_auc = 0.0
    if y_prob is not None:
        try:
            if y_prob.shape[1] == 4:
                roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except Exception:
            roc_auc = 0.0

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "confusion_matrix": cm
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str = None):
    """Plot publication-quality normalized confusion matrix heatmap."""
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax
    )

    ax.set_title("Normalized Confusion Matrix — TrustOCT Proposed Model", fontsize=11, fontweight='bold', pad=12)
    ax.set_ylabel("True Retinal Category", fontsize=10)
    ax.set_xlabel("Predicted Retinal Category", fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
