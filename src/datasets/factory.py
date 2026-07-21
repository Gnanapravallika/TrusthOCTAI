"""Dataset and DataLoader factory module for TrustOCT framework."""

import os
import sys
from typing import Dict, Tuple
import yaml
from torch.utils.data import DataLoader

try:
    from src.datasets.kermany import KermanyDataset
    from src.datasets.loader import create_dataloader
    from src.preprocessing.transforms import get_train_transforms, get_val_transforms
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.datasets.kermany import KermanyDataset
    from src.datasets.loader import create_dataloader
    from src.preprocessing.transforms import get_train_transforms, get_val_transforms


def load_config(path: str) -> dict:
    """Helper to load a YAML config file.

    Args:
        path: Path to YAML file.

    Returns:
        Dict representing YAML content.
    """
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_dataset_and_loader(
    split: str,
    dataset_config_path: str,
    augmentation_config_path: str
) -> Tuple[KermanyDataset, DataLoader]:
    """Dynamically construct dataset split and its DataLoader from configs.

    Args:
        split: Split identifier ('train', 'val', or 'test').
        dataset_config_path: Path to dataset configuration YAML.
        augmentation_config_path: Path to augmentation configuration YAML.

    Returns:
        Tuple of (DatasetInstance, DataLoaderInstance).
    """
    dataset_cfg = load_config(dataset_config_path)
    aug_cfg = load_config(augmentation_config_path)

    # Combined configuration for transform generation
    combined_cfg = {**dataset_cfg, **aug_cfg}

    # Fetch configuration parameters
    dataset_name = dataset_cfg.get("dataset", {}).get("name", "Kermany OCT2017")
    paths = dataset_cfg.get("paths", {})
    loader_cfg = dataset_cfg.get("loader", {})

    # Select split path
    split_path = paths.get(split)
    if not split_path:
        raise ValueError(f"Split '{split}' path is not defined in dataset config.")

    # Instantiate transforms
    if split == "train":
        transforms = get_train_transforms(combined_cfg)
        shuffle = loader_cfg.get("shuffle", True)
    else:
        transforms = get_val_transforms(combined_cfg)
        shuffle = False

    # Instantiate dataset (supporting Kermany initially)
    if "kermany" in dataset_name.lower():
        dataset = KermanyDataset(base_path=split_path, transform=transforms)
    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' factory not yet implemented.")

    # Create loader
    loader = create_dataloader(
        dataset=dataset,
        batch_size=loader_cfg.get("batch_size", 32),
        num_workers=loader_cfg.get("num_workers", 4),
        pin_memory=loader_cfg.get("pin_memory", True),
        shuffle=shuffle,
        drop_last=(split == "train")  # drop last only during training to avoid small batches
    )

    return dataset, loader
