"""Validator loop module for validating TrustOCT models."""

from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def validate_model(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    head_type: str = "edl",
    epoch: int = 0
) -> Tuple[float, float]:
    """Validate model over val_loader split."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            if head_type.lower() == "edl":
                _, alpha = outputs
                loss = criterion(alpha, targets, epoch)
                preds = torch.argmax(alpha, dim=1)
            else:
                loss = criterion(outputs, targets)
                preds = torch.argmax(outputs, dim=1)

            running_loss += loss.item() * images.size(0)
            correct += (preds == targets).sum().item()
            total += images.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc
