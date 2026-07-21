"""Reliability diagram plotting script for TrustOCT framework."""

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_bins: int = 10,
    save_path: str = "reliability_diagram.png"
) -> None:
    """Generate and save a publication-quality Reliability Diagram with ECE.

    Args:
        y_true: Ground truth class indices [N].
        y_prob: Predicted class probabilities [N, K].
        num_bins: Partition bins count.
        save_path: Path to save the generated figure.
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_accs = []
    bin_confs = []
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
            bin_confs.append(avg_confidence_in_bin)
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)

    # Plotting style
    plt.figure(figsize=(6, 6), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    
    # Plot bins
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

    # Gap indicators
    for i in range(num_bins):
        if bin_sizes[i] > 0:
            plt.plot(
                [bin_centers[i], bin_centers[i]],
                [bin_accs[i], bin_centers[i]],
                color="red",
                linestyle="-",
                linewidth=1.5
            )

    plt.xlabel("Confidence", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(f"Reliability Diagram (ECE = {ece:.4f})", fontsize=14, fontweight="bold")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Reliability diagram saved to: {save_path}")
