# TrustOCT: Research Plan

## 1. Literature & References
Our framework is positioned directly to address translation and validation gaps in Optical Coherence Tomography (OCT) deep learning. The primary references that justify our project decisions are:

- **OCT Standards & Evolution**:
  - Fujimoto et al. (2016) - General clinical utility of OCT.
  - Everett et al. (2020), Chong et al. (2024) - Establishing OCT as the gold standard in modern clinical ophthalmology.
- **Workflow & Access Obstacles**:
  - Laíns et al. (2021), Ang et al. (2018), Ciarmatori et al. (2023) - Highlight barriers to widespread swept-source and OCTA adoption, including interpretation and acquisition cost.
  - Chopra et al. (2020), Devine et al. (2025), Zeppieri et al. (2023) - Identify clinic-bound devices as major bottlenecks and motivate remote/portable surveillance.
- **Deployment & Reliability Gaps (Our Core Focus)**:
  - Li et al. (2023) - Identifies clinical deployment, scarce public datasets, cross-site performance drop, and transparency as the dominant research bottlenecks (rather than proof-of-concept classification accuracy).
  - Leitgeb et al. (2021) - Discusses the need for model robustness and validation.

---

## 2. Dataset Strategy & Verification
We test our three claims by organizing our datasets into training (source) and external evaluation (target) domains:

| Dataset Name | Domain Role | Manufacturer / Device | Image Characteristics | Quantity |
| :--- | :--- | :--- | :--- | :--- |
| **Kermany OCT2017** | Source Domain (Train/Val) | Heidelberg Spectralis | Spectral-domain, high SNR | ~84,000 |
| **OCTDL** | Target Domain 1 (Unseen Test) | Various scanner brands | Diverse acquisition protocols | ~2,000 |
| **NEH-UT** | Target Domain 2 (Unseen Test) | Topcon / Zeiss | Unique scan quality / contrast | ~1,500 |
| **OCTID** (Optional) | Target Domain 3 (Unseen Test) | Heidelberg / Zeiss | Focuses on foveal abnormalities | ~500 |

---

## 3. Core Contributions & Novelty
- Instead of proposing a novel convolutional backbone, we explore **how to learn scanner-invariant features** in a single-source setting.
- We integrate **evidential classification** (predicting parameters of a Dirichlet distribution) rather than utilizing standard post-hoc temperature scaling, providing better calibration on unseen scanner domains.
- We evaluate a **selective prediction framework** where the model chooses to defer low-confidence cases, demonstrating the performance-risk trade-offs for clinical settings.

---

## 4. Project Milestones
- **Milestone 1**: Local repository skeleton setup and configuration freeze (Today).
- **Milestone 2**: Preprocessing and dataset pipelines verified (Days 2-3).
- **Milestone 3**: Baseline ResNet50 trained on Kermany OCT2017 (Days 4-5).
- **Milestone 4**: Component implementation (CBAM, MultiScale, MixStyle/CORAL, EDL) (Days 6-10).
- **Milestone 5**: Ablation studies and validation evaluations completed (Days 11-14).
- **Milestone 6**: External dataset generalization testing and calibration checks (Days 15-18).
- **Milestone 7**: Explainability analysis (LayerCAM vs Grad-CAM) and paper writing (Days 19-25).
