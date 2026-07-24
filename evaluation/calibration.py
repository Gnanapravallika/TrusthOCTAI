"""Model confidence calibration and reliability evaluation metrics."""

import os
import numpy as np
import matplotlib.pyplot as plt


def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, num_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return float(ece)


def calculate_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Multi-class Brier Score."""
    num_classes = y_prob.shape[1]
    y_true_onehot = np.eye(num_classes)[y_true]
    brier = np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1))
    return float(brier)


def plot_reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray, num_bins: int = 10, save_path: str = None):
    """Plot reliability diagram (Confidence vs Accuracy)."""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_accs = []
    bin_confs = []

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        if np.sum(in_bin) > 0:
            bin_accs.append(np.mean(accuracies[in_bin]))
            bin_confs.append(np.mean(confidences[in_bin]))
        else:
            bin_accs.append(0)
            bin_confs.append((bin_lower + bin_upper) / 2.0)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax.bar(
        bin_boundaries[:-1],
        bin_accs,
        width=1.0 / num_bins,
        align='edge',
        alpha=0.6,
        color='#1f77b4',
        edgecolor='black',
        label='Model Calibration'
    )

    ece = calculate_ece(y_true, y_prob, num_bins)
    ax.set_title(f"Reliability Diagram (ECE = {ece:.4f})", fontsize=11, fontweight='bold')
    ax.set_xlabel("Confidence", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc="upper left")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
