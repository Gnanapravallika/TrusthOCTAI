"""Testing and inference evaluation engine."""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def test_model(model: torch.nn.Module, test_loader: DataLoader, device: torch.device, head_type: str = "softmax"):
    """
    Run inference over test set.
    Returns:
        y_true: np.ndarray (N,)
        y_pred: np.ndarray (N,)
        y_prob: np.ndarray (N, C)
    """
    model.eval()
    model = model.to(device)

    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    for images, targets in tqdm(test_loader, desc="Testing", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        if head_type == "edl":
            evidence, alpha = model(images)
            S = torch.sum(alpha, dim=1, keepdim=True)
            probs = alpha / S
        else:
            logits = model(images)
            probs = F.softmax(logits, dim=1)

        preds = torch.argmax(probs, dim=1)

        all_y_true.extend(targets.cpu().numpy())
        all_y_pred.extend(preds.cpu().numpy())
        all_y_prob.extend(probs.cpu().numpy())

    return np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob)
