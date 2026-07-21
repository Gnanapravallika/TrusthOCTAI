# Methodology: TrustOCT Framework

This document outlines the detailed scientific and engineering methodology for the **TrustOCT** framework. It serves as our design blueprint and will be updated as we analyze and justify each technical choice.

---

## 1. Research Workflow
Our research follows a systematic path from clinical needs to reproducible validation:

```
[Clinical Gaps] ──> [Research Questions] ──> [Method Selection & Justification]
                                                            │
                                                            ▼
[Rigorous Evaluation] <── [Ablation Experiments] <── [Modular Code & Run]
```

---

## 2. Dataset Selection
To evaluate generalization under domain shift, we partition our data as follows:
- **Source Domain (Train & Validation)**: Kermany OCT2017.
- **Target Domains (Unseen External Testing)**: OCTDL, NEH-UT, and (optional) OCTID.

---

## 3. Data Preprocessing
To address image quality differences between scanners, we establish a two-stage preprocessing pipeline:
1. **Denoising**: Speckle noise reduction using a bilateral filter (retains sharp retinal layer boundaries).
2. **Contrast Enhancement**: Contrast Limited Adaptive Histogram Equalization (CLAHE) to standardize pixel intensity profiles across different manufacturers.

---

## 4. Baseline Model
Our baseline model establishes standard clinical diagnostic performance without advanced trust features:
- **Architecture**: A standard convolutional network (ResNet50) or transformer (ConvNeXt-Tiny).
- **Optimization**: Standard classification using Cross-Entropy loss.

---

## 5. Proposed Framework
The complete TrustOCT architecture integrates multiple swappable components:
- Pretrained backbone.
- Multi-scale feature fusion layers.
- Spatial/Channel attention block (e.g., CBAM).
- Domain generalization module.
- Evidential classification head.

---

## 6. Domain Generalization (DG)
*Under Review & Selection.*
We must justify our selection among:
- **MixStyle**: Feature statistic perturbation.
- **Deep CORAL**: Second-order statistic covariance alignment.
- **DANN / AdDANN**: Adversarial domain adaptation.
- **MMD (Maximum Mean Discrepancy)**: Distance-based feature alignment.
- **DSU (Domain Randomization)**: Feature uncertainty modeling.

---

## 7. Uncertainty Estimation
*Under Review & Selection.*
We must justify our selection among:
- **Evidential Deep Learning (EDL)**: Dirichlet parameterization.
- **Monte Carlo (MC) Dropout**: Test-time statistical variance.
- **Deep Ensembles**: Prediction averaging.

---

## 8. Explainability (XAI)
*Under Review & Selection.*
We must justify our selection among:
- **LayerCAM**: Layer-specific activation mapping.
- **Grad-CAM**: Gradient-based visual attribution.
- **Grad-CAM++**: Second-order derivative visualization.

---

## 9. Loss Functions
Depending on selections in Sections 6 and 7, the total loss function $\mathcal{L}_{total}$ is formulated as:
\[ \mathcal{L}_{total} = \mathcal{L}_{cls} + \lambda_{dg} \mathcal{L}_{dg} + \lambda_{reg} \mathcal{L}_{reg} \]

---

## 10. Training Strategy
- Optimized with learning rate schedulers and checkpoints.
- Early stopping monitored via validation ECE and validation loss.

---

## 11. Evaluation Strategy
We track three clinical diagnostic claims:
1. **Generalization Accuracy**: Cross-dataset macro metrics.
2. **Calibration Quality**: ECE and Reliability Diagrams.
3. **Selective Prediction Risk**: Deferral accuracy curves.

---

## 12. Ablation Studies
Systematic removal of each component to quantify its exact contribution:
- Baseline vs. +Multi-Scale vs. +CBAM vs. +DG vs. +EDL.
