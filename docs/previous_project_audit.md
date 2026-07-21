# Audit Report: Critique of Previous Project (AE-ResNet) & Blueprint for TrustOCT

This report performs a rigorous audit of the previous **AE-ResNet** project's experimental results (Run 7 / SEM 4 PPT) from an M.Tech External Reviewer and journal reviewer perspective. It outlines the red flags, computational bugs, and methodological gaps in the previous codebase, and defines how **TrustOCT** resolves them to ensure publication success.

---

## 1. Key Red Flags in Previous Project Results

### Red Flag A: The Target Generalization Collapse (AUC = 0.496)
In the cross-scanner evaluation on the **OCTID** dataset (Table 5), the proposed **AE-ResNet** model collapsed, achieving:
- **Accuracy**: 36.75% (in one run) and **19.73%** (in another).
- **ROC-AUC**: **0.4962** (random guessing).
- **Comparison**: Vanilla DenseNet-121 achieved **57.83%** and ResNet-50 achieved **42.36%**.

> [!CAUTION]
> **Reviewer Verdict**: A proposed architecture that performs significantly *worse* than the vanilla baselines (and equivalent to a coin toss) invalidates the main claim of the paper. This indicates that the proposed modules (AMSF/CSA) overfitted to the training scanner's noise profile.

### Red Flag B: Target Accuracy of Exactly 0.00%
In Table 5.5 (Cross-Domain Decay), several runs printed **0.00% Target Accuracy** and **-85.25% Absolute Drop** for all models.
- **Root Cause**: This is a coding/data bug. A target accuracy of exactly 0.00% suggests that either the data loader failed to fetch images, the index mapping of classes was offset, or the script encountered an unhandled exception during target validation.

### Red Flag C: Statistically Insignificant Gains (p >= 0.05)
The McNemar test comparing AE-ResNet v2 against DenseNet-121 and ResNet-50 reported:
- **p-value (vs. DenseNet-121)**: 0.29129
- **p-value (vs. ResNet + AMSF)**: 0.64761
- **Verdict**: ❌ *Difference is not statistically significant.*

> [!WARNING]
> **Reviewer Verdict**: Even if AE-ResNet showed slightly different numbers, the statistical test proves that this change is likely due to random variance rather than architectural superiority.

### Red Flag D: Negative Component Ablation (Table 3)
- **ResNet + AMSF (Learnable)**: **87.25%** Accuracy.
- **AE-ResNet v2 (Proposed, AMSF + CSA)**: **86.50%** Accuracy.
- **Verdict**: Adding the Channel-Spatial Attention (CSA) module *reduced* the model's accuracy by **0.75%**. This makes it impossible to justify the inclusion of CSA to reviewers.

---

## 2. Why Did the Previous Model Fail under Domain Shift?
1. **Tiny Training Cohort**: Training a deep model from scratch on the small **OCTDL** dataset (~2,000 images) causes the network to memorize scanner-specific artifacts (contrast, brightness, noise pattern) instead of clinical retinal layers.
2. **Label Mismatch Collisions**: The source dataset (OCTDL) has **7 classes** (AMD, DME, ERM, RAO, RVO, VID, Normal). The target dataset (OCTID) has **5 classes** (AMD, Normal, CSR, DR, MH). Directly predicting a 7-class distribution on a 5-class target without explicit class-overlap mapping causes predictions to fail.
3. **No Domain Generalization (DG) Objectives**: The previous architecture used attention gates (CSA) but did not feature style-mixing or feature covariance-matching loss (CORAL), meaning it had no mathematical incentive to align feature distributions between scanners.

---

## 3. How TrustOCT Corrects These Gaps (V3.0 Blueprint)

```
       Old Pipeline (AE-ResNet)                    Proposed TrustOCT Pipeline
  
  ImageNet (2D General Features)             ImageNet Pretraining (2D Features)
               │                                            │
               ▼                                            ▼
   Direct Train on OCTDL (~2k)                OCT2017 Pretraining (84k Retinal Features)
 (Overfits to Cirrus HD scanner)              (Learns general pathology & anatomy)
               │                                            │
               ▼                                            ▼
   Direct Test on OCTID (5 classes)         Transfer to OCTDL with MixStyle / CORAL
       (Generalization collapse)              (Minimizes scanner-specific domain shift)
                                                            │
                                                            ▼
                                              Uncertainty-Aware Selective Prediction
                                              (Abstains and defers low-confidence scans)
```

| Problem in AE-ResNet | Solution in TrustOCT | Mathematical/Engineering Rationale |
| :--- | :--- | :--- |
| **Overfitting on Small Data** | **OCT2017 Pre-training** | Pre-train on the massive **Kermany OCT2017** dataset (~84,000 images) first. The model learns robust representation of retinal layers before fine-tuning on OCTDL. |
| **Scanner Domain Shift** | **MixStyle & Feature CORAL** | Explicitly mix feature statistics (mean/variance) during training to enforce scanner-invariance. |
| **Label Incompatibility** | **Explicit Overlap Mapping** | The evaluation module (`src/evaluation/cross_dataset.py`) will filter target evaluations to mutually overlapping classes (AMD, Normal) or compute OOD detection scores. |
| **Overconfident Errors** | **Evidential Deep Learning** | Replaces standard softmax with a Dirichlet distribution head, allowing the model to flag unfamiliar scans as high uncertainty. |
