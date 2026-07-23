"""Consolidated dataset loaders, verification utilities, and statistics calculations for TrustOCT."""

import json
import os
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union
import yaml
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

# =====================================================================
# 1. Constants
# =====================================================================

CLASS_NAMES: List[str] = ["CNV", "DME", "DRUSEN", "NORMAL"]
CLASS_TO_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".jpeg", ".jpg", ".png")
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (224, 224)


# =====================================================================
# 2. Dataset Classes
# =====================================================================

class BaseOCTDataset(Dataset, ABC):
    """Abstract Base Class for all OCT datasets."""

    def __init__(
        self,
        base_path: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        self.base_path = base_path
        self.transform = transform
        self.target_transform = target_transform
        
        self.filepaths: List[str] = []
        self.labels: List[int] = []
        self._load_metadata()

    @abstractmethod
    def _load_metadata(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath = self.filepaths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(filepath).convert("RGB")
            image_np = np.array(image)
        except Exception as e:
            raise RuntimeError(f"Error loading image {filepath}: {e}")

        if self.transform:
            augmented = self.transform(image=image_np)
            image_tensor = augmented["image"]
        else:
            image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float() / 255.0

        if self.target_transform:
            label = self.target_transform(label)

        return image_tensor, label


class KermanyDataset(BaseOCTDataset):
    """Kermany OCT2017 Dataset loader implementation."""

    def _load_metadata(self) -> None:
        if not os.path.exists(self.base_path):
            raise FileNotFoundError(f"Kermany path does not exist at {self.base_path}")

        for class_name in CLASS_NAMES:
            class_dir = os.path.join(self.base_path, class_name)
            if not os.path.exists(class_dir):
                continue

            for filename in os.listdir(class_dir):
                file_path = os.path.join(class_dir, filename)
                if os.path.isdir(file_path):
                    continue

                _, ext = os.path.splitext(filename)
                if ext.lower() in SUPPORTED_EXTENSIONS:
                    self.filepaths.append(file_path)
                    self.labels.append(CLASS_TO_INDEX[class_name])

        print(f"Loaded Kermany split at {self.base_path} with {len(self.filepaths)} images.")


# =====================================================================
# 3. Loader & Factory
# =====================================================================

def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """Create a standard PyTorch DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        drop_last=drop_last
    )


def get_dataset_and_loader(
    split: str,
    dataset_config_path: str,
    augmentation_config_path: str
) -> Tuple[KermanyDataset, DataLoader]:
    """Factory loading transforms and datasets from YAML configs."""
    # Lazy imports to avoid circular dependencies
    from datasets.transforms import get_train_transforms, get_val_transforms

    with open(dataset_config_path, "r") as f:
        dataset_cfg = yaml.safe_load(f)
    with open(augmentation_config_path, "r") as f:
        aug_cfg = yaml.safe_load(f)

    combined_cfg = {**dataset_cfg, **aug_cfg}

    dataset_name = dataset_cfg.get("dataset", {}).get("name", "Kermany OCT2017")
    paths = dataset_cfg.get("paths", {})
    loader_cfg = dataset_cfg.get("loader", {})

    split_path = paths.get(split)
    if not split_path:
        raise ValueError(f"Split '{split}' path is not defined in dataset config.")

    if split == "train":
        transforms = get_train_transforms(combined_cfg)
        shuffle = loader_cfg.get("shuffle", True)
    else:
        transforms = get_val_transforms(combined_cfg)
        shuffle = False

    if "kermany" in dataset_name.lower():
        dataset = KermanyDataset(base_path=split_path, transform=transforms)
    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' not yet supported.")

    loader = create_dataloader(
        dataset=dataset,
        batch_size=loader_cfg.get("batch_size", 32),
        num_workers=loader_cfg.get("num_workers", 4),
        pin_memory=loader_cfg.get("pin_memory", True),
        shuffle=shuffle,
        drop_last=(split == "train")
    )

    return dataset, loader
