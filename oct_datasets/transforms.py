"""Bilateral filter, CLAHE and normalization transformations for TrustOCT.

Training pipeline (8 steps):
    1. Bilateral Filter    — OCT speckle noise removal (edge-preserving)
    2. CLAHE               — Retinal layer contrast enhancement
    3. Resize (224x224)    — Required for ResNet50 input
    4. Random Rotation     — ±15°, simulates patient positioning variation
    5. Horizontal Flip     — Simulates left/right retinal scan variation
    6. Brightness/Contrast — Simulates illumination variability across scanners
    7. Gaussian Noise      — Simulates OCT scanner noise variability (p=0.2)
    8. ImageNet Normalize  — Match pretrained ResNet50 statistics

Validation/Test pipeline (4 steps, deterministic):
    1. Bilateral Filter    — Same as training for consistency
    2. CLAHE               — Same as training for consistency
    3. Resize (224x224)    — Required for ResNet50 input
    4. ImageNet Normalize  — Match pretrained ResNet50 statistics

Note: Elastic Distortion, Grid Distortion, Coarse Dropout, and Random Crop
are intentionally excluded. These create anatomically implausible retinal
images and cannot be clinically justified for OCT B-scan classification.
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform


class BilateralFilter(ImageOnlyTransform):
    """Custom Albumentations wrapper for OpenCV Bilateral Filter.

    Reduces OCT speckle noise while preserving sharp retinal layer edges.
    This is an edge-preserving smoothing filter — unlike Gaussian blur,
    it does not blur across layer boundaries.
    """

    def __init__(
        self,
        d: int = 9,
        sigma_color: float = 75.0,
        sigma_space: float = 75.0,
        always_apply: bool = False,
        p: float = 1.0
    ):
        super().__init__(always_apply, p)
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        if img.dtype != np.uint8:
            img_uint8 = (img * 255.0).astype(np.uint8)
            denoised = cv2.bilateralFilter(img_uint8, self.d, self.sigma_color, self.sigma_space)
            return (denoised / 255.0).astype(np.float32)
        return cv2.bilateralFilter(img, self.d, self.sigma_color, self.sigma_space)

    def get_transform_init_args_names(self):
        return ("d", "sigma_color", "sigma_space")


def get_train_transforms(config: dict) -> A.Compose:
    """Build training transformation pipeline based on YAML configuration.

    Returns a clinically justified 8-step pipeline for OCT B-scan augmentation.
    Each step is evidence-based and reproducible for dissertation documentation.
    """
    preprocessing_cfg = config.get("preprocessing", {})
    augmentations_cfg = config.get("augmentations", {})
    resize_h, resize_w = preprocessing_cfg.get("resize", [224, 224])

    transform_list = []

    # Step 1: Bilateral Denoising
    # Removes OCT speckle noise while preserving retinal layer boundaries.
    bf_cfg = preprocessing_cfg.get("bilateral_filter", {})
    if bf_cfg.get("enabled", True):
        transform_list.append(
            BilateralFilter(
                d=bf_cfg.get("d", 9),
                sigma_color=bf_cfg.get("sigma_color", 75.0),
                sigma_space=bf_cfg.get("sigma_space", 75.0),
                p=1.0
            )
        )

    # Step 2: CLAHE — Contrast Limited Adaptive Histogram Equalization
    # Enhances retinal layer contrast. Widely used in OCT preprocessing literature.
    clahe_cfg = preprocessing_cfg.get("clahe", {})
    if clahe_cfg.get("enabled", True):
        transform_list.append(
            A.CLAHE(
                clip_limit=clahe_cfg.get("clip_limit", 2.0),
                tile_grid_size=tuple(clahe_cfg.get("tile_grid_size", [8, 8])),
                p=1.0
            )
        )

    # Step 3: Resize to 224x224 — required for ResNet50 input dimensions
    transform_list.append(A.Resize(height=resize_h, width=resize_w))

    # Step 4: Random Rotation (±15°)
    # Simulates slight variations in patient head positioning during OCT scan.
    rot_cfg = augmentations_cfg.get("random_rotate", {})
    transform_list.append(
        A.Rotate(limit=rot_cfg.get("limit", 15), p=rot_cfg.get("p", 0.5))
    )

    # Step 5: Horizontal Flip
    # Simulates scanning the left vs right eye (mirror-image retinal anatomy).
    hf_cfg = augmentations_cfg.get("horizontal_flip", {})
    transform_list.append(A.HorizontalFlip(p=hf_cfg.get("p", 0.5)))

    # Step 6: Random Brightness/Contrast
    # Simulates illumination variability across different OCT scanner models.
    bc_cfg = augmentations_cfg.get("random_brightness_contrast", {})
    transform_list.append(
        A.RandomBrightnessContrast(
            brightness_limit=bc_cfg.get("brightness_limit", 0.1),
            contrast_limit=bc_cfg.get("contrast_limit", 0.1),
            p=bc_cfg.get("p", 0.5)
        )
    )

    # Step 7: Gaussian Noise (p=0.2)
    # Simulates varying sensor noise from different OCT scanner manufacturers.
    # Low probability used intentionally to avoid degrading meaningful scan features.
    # Elastic/Grid Distortion, Coarse Dropout, and Random Crop are intentionally
    # excluded as they create anatomically implausible retinal structures.
    transform_list.append(
        A.GaussNoise(std_range=(0.02, 0.11), p=0.2)
    )


    # Step 8: ImageNet Normalization
    # Required to match the statistical distribution of the ResNet50 pretrained weights.
    norm_cfg = augmentations_cfg.get("normalize", {})
    transform_list.append(
        A.Normalize(
            mean=norm_cfg.get("mean", [0.485, 0.456, 0.406]),
            std=norm_cfg.get("std", [0.229, 0.224, 0.225]),
            p=1.0
        )
    )

    # Tensor conversion — convert NumPy HWC array to PyTorch CHW tensor
    transform_list.append(ToTensorV2())

    return A.Compose(transform_list)


def get_val_transforms(config: dict) -> A.Compose:
    """Build validation/testing transformation pipeline.

    Deterministic 4-step pipeline — no stochastic augmentations.
    This ensures reproducible, consistent evaluation reflecting real deployment.
    """
    preprocessing_cfg = config.get("preprocessing", {})
    augmentations_cfg = config.get("augmentations", {})
    resize_h, resize_w = preprocessing_cfg.get("resize", [224, 224])

    transform_list = []

    # Step 1: Bilateral Denoising (same as training for consistency)
    bf_cfg = preprocessing_cfg.get("bilateral_filter", {})
    if bf_cfg.get("enabled", True):
        transform_list.append(
            BilateralFilter(
                d=bf_cfg.get("d", 9),
                sigma_color=bf_cfg.get("sigma_color", 75.0),
                sigma_space=bf_cfg.get("sigma_space", 75.0),
                p=1.0
            )
        )

    # Step 2: CLAHE (same as training for consistency)
    clahe_cfg = preprocessing_cfg.get("clahe", {})
    if clahe_cfg.get("enabled", True):
        transform_list.append(
            A.CLAHE(
                clip_limit=clahe_cfg.get("clip_limit", 2.0),
                tile_grid_size=tuple(clahe_cfg.get("tile_grid_size", [8, 8])),
                p=1.0
            )
        )

    # Step 3: Resize to 224x224
    transform_list.append(A.Resize(height=resize_h, width=resize_w))

    # Step 4: ImageNet Normalization
    norm_cfg = augmentations_cfg.get("normalize", {})
    transform_list.append(
        A.Normalize(
            mean=norm_cfg.get("mean", [0.485, 0.456, 0.406]),
            std=norm_cfg.get("std", [0.229, 0.224, 0.225]),
            p=1.0
        )
    )

    # Tensor conversion
    transform_list.append(ToTensorV2())

    return A.Compose(transform_list)
