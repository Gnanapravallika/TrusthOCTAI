"""Confusion matrix plotting module for TrustOCT framework."""

import os
from typing import List
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    cm: list,
    classes: List[str],
    save_path: str = "confusion_matrix.png",
    normalize: bool = False
) -> None:
    """Generate and save a publication-quality confusion matrix.

    Args:
        cm: Confusion matrix list of lists.
        classes: List of class category names.
        save_path: Path to save the generated image.
        normalize: Set True to display percentages instead of counts.
    """
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

    # Threshold for text coloring
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

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")
