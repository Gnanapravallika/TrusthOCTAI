"""DataLoader modules for TrustOCT framework."""

import os
import sys
from typing import Dict, Optional
from torch.utils.data import DataLoader, Dataset

try:
    from src.datasets.constants import CLASS_NAMES
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.datasets.constants import CLASS_NAMES


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """Create a standard PyTorch DataLoader.

    Args:
        dataset: A PyTorch Dataset instance.
        batch_size: Size of input batches.
        num_workers: Number of CPU worker threads.
        pin_memory: Allocate page-locked memory for fast GPU transfers.
        shuffle: Shuffle dataset elements.
        drop_last: Drop the final batch if it's incomplete.

    Returns:
        PyTorch DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        drop_last=drop_last
    )
