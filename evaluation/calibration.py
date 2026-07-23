"""Calibration metrics (ECE and Brier score) and reliability diagram plotting for TrustOCT."""

import os
import numpy as np
import matplotlib.pyplot as plt


def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, num_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE)."""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)

    ece = 0.0
    num_samples = len(y_true)
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return float(ece)


def calculate_brier_score(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int = 4) -> float:
    """Compute multi-class Brier Score."""
    n_samples = len(y_true)
    if n_samples == 0:
        return 0.0
    y_one_hot = np.zeros((n_samples, num_classes))
    y_one_hot[np.arange(n_samples), y_true] = 1.0
    brier = np.sum((y_prob - y_one_hot) ** 2) / n_samples
    return float(brier)


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_bins: int = 10,
    save_path: str = "reliability_diagram.png"
) -> None:
    """Generate and save ECE reliability diagram."""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_accs = []
    bin_sizes = []

    ece = 0.0
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        bin_sizes.append(prop_in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            bin_accs.append(accuracy_in_bin)
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
        else:
            bin_accs.append(0.0)

    plt.figure(figsize=(6, 6), dpi=300)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    plt.bar(
        bin_centers,
        bin_accs,
        width=1.0 / num_bins,
        edgecolor="black",
        color="#1f77b4",
        alpha=0.8,
        label="Accuracy"
    )

    for i in range(num_bins):
        if bin_sizes[i] > 0:
            plt.plot(
                [bin_centers[i], bin_centers[i]],
                [bin_accs[i], bin_centers[i]],
                color="red",
                linestyle="-"
            )

    plt.xlabel("Confidence", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(f"Reliability Diagram (ECE = {ece:.4f})", fontsize=14, fontweight="bold")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="upper left")
    plt.tight_layout()
    
    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Reliability diagram saved to: {save_path}")
