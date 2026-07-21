# TrustOCT: Implementation Plan

This implementation plan details how the codebase is structured for dynamic component swappability using a configuration registry.

---

## 1. Registry & Module Builder (`src/registry/`)
We use a builder-pattern registry to build the model dynamically based on configuration parameters:

- **Registry (`src/registry/builder.py`)**:
  - Dynamically registers and fetches model components.
  - Exposes `build_model(config)` to construct the `TrustOCT` network.
  - Exposes `build_loss(config)` to construct the classification, CORAL, or Evidential losses.
  - Exposes `build_datasets(config)` to create dataloaders.

---

## 2. Decoupled Models (`src/models/`)
The architecture is structured into standalone modular scripts under `src/models/`:

- **Backbone (`src/models/backbone.py`)**:
  - Encapsulates pre-trained CNN/Transformer architectures (e.g., `resnet50`, `convnext_tiny`).
  - Exposes intermediate layers (e.g., Layer 3 and Layer 4 of ResNet50) to support multi-scale feature fusion.
- **Multi-Scale Fusion (`src/models/multiscale.py`)**:
  - Performs spatial upsampling and feature concatenation:
    - Upsamples Layer 3 features to match Layer 4 resolution.
    - Concatenates channels: `[Batch, 1024, 7, 7] + [Batch, 2048, 7, 7] = [Batch, 3072, 7, 7]`.
    - Reduces dimensionality via a Conv2d block to output `[Batch, 512, 7, 7]`.
- **Attention Block (`src/models/cbam.py`)**:
  - Implements the Convolutional Block Attention Module (Channel Attention followed by Spatial Attention).
- **Heads (`src/models/heads/`)**:
  - `base_head.py`: Abstract class for prediction heads.
  - `softmax.py`: Standard classification head mapping features to $K$ disease logits.
  - `edl.py`: Dirichlet classification head producing positive values ($\alpha_k > 0$) mapping to Dirichlet parameters.
- **TrustOCT Model (`src/models/trustoct.py`)**:
  - The orchestrator class (`TrustOCT`) that ties the backbone, feature module, attention block, and head together based on configuration toggles.

---

## 3. Domain Generalization (`src/domain_generalization/`)
- `identity.py`: Passes features through unchanged.
- `mixstyle.py`: Computes channel mean and variance statistics for random pairs of features in a batch and mixes them using a beta distribution.
- `coral.py`: Computes covariance matrices and aligns them across source domains.

---

## 4. Preprocessing & Dataloaders (`src/preprocessing/` & `src/datasets/`)
- **Transforms (`src/preprocessing/transforms.py`)**:
  - Combines OpenCV Bilateral filter (for speckle reduction) and CLAHE (for contrast optimization) with Albumentations.
- **Dataset Factory (`src/datasets/factory.py`)**:
  - Dynamically builds Kermany, OCTDL, or NEH-UT datasets using custom config keys.

---

## 5. Development Milestones
1. Configure dependencies and verify folder layout.
2. Build data pipelines (Bilateral + CLAHE preprocessing + custom dataloaders).
3. Build model builder and baseline ResNet50 classifier.
4. Implement loss functions and standard training loop.
5. Add swappable modules: CBAM, MixStyle, and Evidential heads.
6. Verify pipeline with automated tests.
