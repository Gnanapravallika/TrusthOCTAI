# TrustOCT: Experiment Plan

This plan documents the configuration and tracking layout for all ablation and comparison experiments needed for the paper.

---

## 1. Structured Experiments Directory
Every experiment is run using a specific configuration file. All results, logs, checks, and checkpoints are stored in isolated versioned folders:

```
experiments/
├── EXP001_Baseline/
│   ├── config.yaml
│   ├── metrics.json
│   ├── weights.pth
│   ├── plots/
│   └── log.txt
├── EXP002_CBAM/
├── EXP003_MultiScale/
├── EXP004_MixStyle/
├── EXP005_CORAL/
├── EXP006_EDL/
└── EXP007_TrustOCT/ (Proposed)
```

---

## 2. Core Experiments & Ablations

| Exp ID | Model Architecture Description | Target Loss Function | Claims Evaluated |
| :--- | :--- | :--- | :--- |
| **EXP001** | **Baseline**: Standard ResNet50 | Standard Cross-Entropy | Reference baseline. |
| **EXP002** | ResNet50 + CBAM Attention | Standard Cross-Entropy | Effect of spatial/channel attention. |
| **EXP003** | ResNet50 + Multi-Scale Feature Fusion + CBAM | Standard Cross-Entropy | Effect of multiscale details. |
| **EXP004** | ResNet50 + MixStyle Generalization | Standard Cross-Entropy | **Claim 1**: MixStyle cross-dataset generalization. |
| **EXP005** | ResNet50 + CORAL Feature Alignment | Cross-Entropy + CORAL loss | Baseline comparison against MixStyle. |
| **EXP006** | ResNet50 + Evidential Head (EDL) | Evidential loss | **Claim 2**: Uncertainty calibration. |
| **EXP007** | **TrustOCT** (Proposed): ResNet50 + MultiScale + CBAM + MixStyle + EDL Head | Evidential loss | **Claim 1, 2, 3**: Overall framework evaluation. |

---

## 3. Evaluation Pipelines

### A. Generalization Metrics (Claim 1)
For each experiment, after training on Kermany, we evaluate the models on unseen target datasets:
- **Metrics**: Macro-F1, Accuracy, Sensitivity, Specificity, and AUC-ROC.
- **Goal**: Demonstrate that MixStyle/CORAL maintains high classification accuracy under device shift compared to standard transfer learning.

### B. Uncertainty Calibration (Claim 2)
We measure how well the prediction confidence correlates with the actual probability of correctness:
- **Metrics**: Expected Calibration Error (ECE), Brier Score, and Reliability Diagrams (confidence vs. accuracy bins).
- **OOD Detection**: Evaluate the model's ability to detect out-of-distribution inputs (e.g., non-eye noise or corrupted scans) using AUROC based on evidential uncertainty.

### C. Selective Prediction (Claim 3)
We simulate clinical deployment by deferring predictions with high uncertainty:
- **Selective Risk (Error)** vs **Coverage**: Plot accuracy of retained predictions as we vary the percentage of deferred cases.
- **Goal**: Show that by deferring the most uncertain 10-20% of scans, our model achieves near-perfect diagnostic accuracy on the remaining clinical cohort.
