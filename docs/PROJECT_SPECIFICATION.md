# TrustOCT: Project Specification

## 1. Introduction
Optical Coherence Tomography (OCT) is a standard non-invasive imaging modality in ophthalmology used to detect macular diseases such as Choroidal Neovascularization (CNV), Diabetic Macular Edema (DME), and Drusen. While deep learning models have achieved expert-level performance on specific datasets, deploying them safely in clinical environments remains a challenge due to dataset shift and overconfident misclassifications.

**TrustOCT** is a deep learning framework designed for trustworthy, cross-device retinal OCT classification. It integrates domain generalization to handle cross-scanner variability and uncertainty-aware selective prediction to identify and defer unreliable/low-quality scans.

---

## 2. Research Gaps
1. **Cross-Device Performance Drops**: Models trained on single-scanner datasets show significant performance drops when evaluated on images from different manufacturers (e.g., Heidelberg Spectralis vs. Zeiss Cirrus) due to variance in contrast, resolution, and noise.
2. **Uncalibrated Confidence**: Standard neural networks output overconfident softmax probabilities even when presented with corrupted, ambiguous, or out-of-distribution (OOD) images, which poses severe clinical risks.
3. **Black-box Explainability**: Common visual attribution methods (like Grad-CAM) often produce coarse, unlocalized heatmaps that do not correspond well to fine clinical lesions like micro-drusen or thin fluid layers.

---

## 3. Research Objectives
- **Objective 1**: Develop a robust feature extraction pipeline that generalizes across different scanner manufacturers without requiring target domain data during training.
- **Objective 2**: Quantify prediction uncertainty using Evidential Deep Learning (EDL) to distinguish between reliable and unreliable predictions.
- **Objective 3**: Implement a selective prediction mechanism that defers low-confidence, high-uncertainty predictions for manual ophthalmologist review.
- **Objective 4**: Benchmark LayerCAM against Grad-CAM and Grad-CAM++ to determine which visual explanation method provides the most precise clinical lesion localization.

---

## 4. Research Questions
- **RQ1**: Can feature-level domain generalization (MixStyle / Feature CORAL) improve classification performance on unseen external OCT datasets?
- **RQ2**: Can evidential uncertainty estimation reliably identify incorrect predictions and out-of-distribution images?
- **RQ3**: Does selective prediction (abstention under high uncertainty) improve the safety and accuracy of the retained predictions?
- **RQ4**: Does LayerCAM provide superior lesion localization (higher visual alignment with retinal pathology) compared to traditional Grad-CAM on OCT scans?

---

## 5. Proposed Methodology
The TrustOCT framework is built with a LEGO-like modular design comprising:
- **Preprocessing Pipeline**: Bilateral filtering to reduce speckle noise while preserving retinal layers, followed by CLAHE to enhance contrast.
- **Backbone**: Swappable feature extraction backbone (ResNet50 / ConvNeXt-Tiny).
- **Domain Generalization**: Swappable modules (MixStyle, CORAL, or Identity/None) to learn scanner-invariant representations.
- **Multi-Scale Feature Fusion**: Combines spatial detail from intermediate layers (Layer 3) with global context (Layer 4).
- **Attention Block**: Convolutional Block Attention Module (CBAM) to focus on lesion-relevant spatial and channel features.
- **Evidential Classification Head**: A specialized head producing Dirichlet parameters to quantify prediction evidence and uncertainty.

---

## 6. Datasets
- **Source/Training Domain**: Kermany OCT2017 (~84,000 images; Normal, CNV, DME, Drusen).
- **Target/External Testing Domain 1**: OCTDL (Optical Coherence Tomography Dataset for Deep Learning).
- **Target/External Testing Domain 2**: NEH-UT dataset.
- **Target/External Testing Domain 3 (Optional)**: OCTID (OCT Image Database).

---

## 7. Baselines
We evaluate TrustOCT against the following baseline configurations:
1. **Baseline 1 (Standard Transfer Learning)**: ResNet50 trained with standard cross-entropy loss, without attention, domain generalization, or evidential heads.
2. **Baseline 2 (Attention-augmented)**: ResNet50 + CBAM.
3. **Baseline 3 (Domain-generalized)**: ResNet50 + MixStyle.
4. **Baseline 4 (Evidential-only)**: ResNet50 + Evidential Head.
5. **Proposed (TrustOCT)**: ResNet50 + Multi-Scale Fusion + CBAM + MixStyle + Evidential Head.

---

## 8. Expected Contributions
1. A unified, modular framework (**TrustOCT**) combining domain generalization and evidential deep learning for retinal OCT classification.
2. Direct empirical comparisons between feature-level generalization techniques (MixStyle vs. CORAL) for cross-scanner medical image translation.
3. Rigorous validation of selective prediction using accuracy-coverage curves, demonstrating safety improvements in clinical diagnostic pipelines.
4. Comparative visualization studies highlighting LayerCAM's fine-grained lesion localization over Grad-CAM algorithms.
