"""OCT Image preprocessing and clinical data augmentation transforms using Albumentations."""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BilateralFilterTransform(A.ImageOnlyTransform):
    """OpenCV Bilateral Filter for OCT speckle noise reduction with edge preservation."""

    def __init__(self, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0, always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply, p)
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        if img.ndim == 3 and img.shape[2] == 3:
            # Apply bilateral filter per channel or convert to uint8
            filtered = cv2.bilateralFilter(img, self.d, self.sigma_color, self.sigma_space)
        else:
            filtered = cv2.bilateralFilter(img, self.d, self.sigma_color, self.sigma_space)
        return filtered


def get_train_transforms(config: dict = None) -> A.Compose:
    """Build clinically defensible training augmentation pipeline."""
    return A.Compose([
        # Step 1: Speckle Noise Removal
        BilateralFilterTransform(d=9, sigma_color=75.0, sigma_space=75.0, p=1.0),

        # Step 2: Layer Contrast Enhancement
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),

        # Step 3: Spatial Resizing to ResNet-50 input size
        A.Resize(224, 224),

        # Step 4: Anatomically safe augmentations
        A.RandomRotate90(p=0.2),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),

        # Step 5: Inter-scanner noise variation simulation (std_range compatible with Albumentations 2.x)
        A.GaussNoise(std_range=(0.02, 0.11), p=0.2),

        # Step 6: ImageNet Normalization
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transforms(config: dict = None) -> A.Compose:
    """Build validation / test preprocessing pipeline (deterministic)."""
    return A.Compose([
        BilateralFilterTransform(d=9, sigma_color=75.0, sigma_space=75.0, p=1.0),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
