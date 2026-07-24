# 👁️ TrustOCT: Trustworthy AI Framework for Retinal OCT Diagnosis

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

A trustworthy, clinically explainable, and reliable Deep Learning framework for automated Retinal Optical Coherence Tomography (OCT) image classification across **CNV**, **DME**, **DRUSEN**, and **NORMAL** conditions.

---

## 🧱 Proposed Peak Architecture (`msf_cbam_resnet50`)

The proposed peak winning architecture combines:
1. **ResNet-50 Dual-Layer Feature Extractor**:
   - `Layer 3` ($x_3 \in \mathbb{R}^{B \times 1024 \times 14 \times 14}$): Captures fine spatial details & intra-retinal fluid boundary micro-structures.
   - `Layer 4` ($x_4 \in \mathbb{R}^{B \times 2048 \times 7 \times 7}$): Captures high-level abstract semantic pathology context.
2. **Multi-Scale Feature Fusion (MSF Module)**:
   - Projects Layer 3 features to 2048 channels with stride 2 ($14 \times 14 \to 7 \times 7$) and fuses via element-wise addition:  
     $$F_{\text{fused}} = \text{ReLU}(\text{BN}(\text{Conv}_{1 \times 1}(x_3))) + x_4 \in \mathbb{R}^{B \times 2048 \times 7 \times 7}$$
3. **Dual Convolutional Block Attention Module (CBAM)**:
   - **Channel Attention**: Refines *WHAT* pathology features are relevant ($r=16$).
   - **Spatial Attention**: Focuses on *WHERE* retinal layer fluid/drusen lesions are located ($7 \times 7$ conv).
4. **Classification Head**:
   - Adaptive Global Average Pooling (`AdaptiveAvgPool2d((1, 1))`) $\to$ Linear Projection (`2048` $\to$ `4` classes).

---

## 🚀 Quick Start (Google Colab)

Open `TrustOCT_Colab.ipynb` in Google Colab to execute the end-to-end pipeline:
1. **Direct Kaggle Download**: Downloads and extracts Kermany OCT2017 dataset (`kaggle datasets download -d paultimothymooney/kermany2017`).
2. **Patient-Level Split**: Performs patient-isolated split via `GroupShuffleSplit` (zero patient overlap across splits).
3. **Training & Evaluation**: Fits `msf_cbam_resnet50`, evaluates Confusion Matrix, Calibration (ECE/Brier score), LayerCAM overlays, and exports results directly to Google Drive.

```bash
python main.py --data_dir /path/to/OCT2017 --model_name msf_cbam_resnet50 --epochs 30
```

---

## 📂 Repository Structure

```
TrusthOCTAI/
├── configs/
│   ├── dataset.yaml
│   ├── model.yaml
│   └── train.yaml
├── oct_datasets/
│   ├── dataset.py        # Patient-level splitting & PyTorch Dataset
│   ├── transforms.py     # Bilateral filtering, CLAHE & augmentations
│   └── utils.py          # Dataset scanner
├── models/
│   ├── resnet50.py       # Dual-layer ResNet-50 backbone
│   ├── msf.py            # Multi-Scale Feature Fusion module
│   ├── cbam.py           # Dual Channel + Spatial Attention
│   └── trustoct.py       # TrustOCT unified assembler
├── engine/
│   ├── losses.py         # CrossEntropy & Evidential Dirichlet Loss
│   ├── trainer.py        # Training & validation loop
│   └── tester.py         # Inference engine
├── evaluation/
│   ├── metrics.py        # Classification metrics & Confusion Matrix
│   ├── calibration.py    # ECE, Brier Score & Reliability Diagrams
│   └── robustness.py     # Perturbation & covariate shift suite
├── explainability/
│   ├── layercam.py       # LayerCAM fine attribution engine
│   └── visualization.py # Side-by-side overlay visualization
├── main.py
├── TrustOCT_Colab.ipynb
└── requirements.txt
```
