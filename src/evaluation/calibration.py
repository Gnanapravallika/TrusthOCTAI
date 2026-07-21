"""Expected Calibration Error (ECE) and Brier Score calculations for TrustOCT."""

import numpy as np
from typing import Dict, List, Tuple


def calculate_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_bins: int = 15
) -> float:
    """Compute the Expected Calibration Error (ECE).

    Args:
        y_true: Ground truth integer array of shape [N].
        y_prob: Prediction probability distribution of shape [N, K].
        num_bins: Number of confidence partitioning bins.

    Returns:
        Scalar ECE score.
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)

    ece = 0.0
    n_samples = len(y_true)
    
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Identify indices in current bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return float(ece)


def calculate_brier_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int = 4
) -> float:
    """Compute the multi-class Brier Score.

    Args:
        y_true: Ground truth integer array of shape [N].
        y_prob: Prediction probability distribution of shape [N, K].
        num_classes: Total class count.

    Returns:
        Scalar Brier Score value (lower is better calibrated).
    """
    n_samples = len(y_true)
    if n_samples == 0:
        return 0.0

    # One-hot encode targets
    y_one_hot = np.zeros((n_samples, num_classes))
    y_one_hot[np.arange(n_samples), y_true] = 1.0

    # Mean squared error over probabilities and one-hot targets
    brier = np.sum((y_prob - y_one_hot) ** 2) / n_samples
    return float(brier)


def generate_calibration_stats(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int = 4,
    num_bins: int = 15
) -> Dict[str, float]:
    """Calculate and package calibration performance.

    Args:
        y_true: Ground truth labels [N].
        y_prob: Predicted class probabilities [N, K].
        num_classes: Number of target classes.
        num_bins: ECE bins parameter.

    Returns:
        Dict of ECE and Brier Score.
    """
    ece = calculate_ece(y_true, y_prob, num_bins=num_bins)
    brier = calculate_brier_score(y_true, y_prob, num_classes=num_classes)
    
    return {
        "ece": ece,
        "brier_score": brier
    }
