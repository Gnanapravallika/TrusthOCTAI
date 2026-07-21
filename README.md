# TrustOCT: A Trustworthy Cross-Device Retinal OCT Classification Framework

TrustOCT is a trustworthy deep learning framework designed for retinal Optical Coherence Tomography (OCT) disease classification. The framework enhances classification accuracy and confidence calibration across different device manufacturers by integrating Adaptive Multi-Scale Feature Fusion, CBAM attention gating, MixStyle domain statistic mixing, and an Evidential Deep Learning (EDL) prediction head.

---

## 🚀 Key Framework Architecture

Our modular framework follows a configuration-driven pipeline allowing full ablation studies:
1. **CNN Backbone**: Extract multi-level representations (ResNet50 intermediate features).
2. **Adaptive Multi-Scale Fusion**: Aligns and fuses deep representations (`Layer3` and `Layer4`) to prevent resolution loss.
3. **CBAM Attention**: Employs spatial and channel attention to focus on pathology regions (e.g., fluid pockets, drusen).
4. **MixStyle statistics blending**: Mixes feature means and standard deviations during training to make representations device-invariant.
5. **Evidential Dirichlet Head (EDL)**: Replaces standard Softmax with a Dirichlet concentration model outputting prediction confidence and epistemic uncertainty.

---

## 📁 Repository Structure

```
TrustOCT/
│
├── configs/                # Experiment YAML configuration files
│   ├── dataset.yaml        # Dataset taxonomy, splits, and loader configs
│   ├── augmentation.yaml   # CLAHE, Bilateral filtering, and flip settings
│   ├── train.yaml          # Hyperparameters, patience, and optimization setting
│   ├── EXP001_Baseline.yaml
│   ├── EXP002_MultiScale.yaml
│   ├── EXP003_CBAM.yaml
│   ├── EXP004_MixStyle.yaml
│   ├── EXP005_EDL.yaml
│   └── EXP006_TrustOCT.yaml
│
├── datasets/               # Raw folders (Kermany, OCTID, NEH_UT)
│
├── notebooks/
│   └── TrustOCT_Setup.ipynb  # Master Colab Orchestrator
│
├── outputs/                # Folder for reports, logs, and checkpoints
│   ├── checkpoints/
│   ├── figures/
│   ├── metrics/
│   └── logs/
│
├── tests/
│   └── verify_compile.py   # Compilation integration check
│
├── src/
│   ├── datasets.py         # File verification, statistic calculations, and Dataset/DataLoader classes
│   ├── preprocessing.py    # Custom CV2 Bilateral filter and CLAHE augmentation pipelines
│   ├── models.py           # Backbone, CBAM, Fusion, MixStyle, and unified assembler
│   ├── heads.py            # Softmax and Evidential Dirichlet prediction heads
│   ├── losses.py           # EDL Dirichlet annealing losses
│   ├── trainer.py          # Trainer fit/validate, checkpointing, and logs
│   ├── evaluation.py       # Macro metrics, ECE calibration, Brier score, and OOD detection
│   ├── explainability.py   # Grad-CAM and LayerCAM attribution overlays
│   └── plots.py            # Reliability curves and confusion matrices
│
├── train.py                # Master training execution script
├── requirements.txt        # Package dependencies list
├── README.md               # Framework documentation
└── .gitignore
```

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<username>/TrustOCT.git
   cd TrustOCT
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Verify compilation**:
   Make sure everything compiles on your machine by running:
   ```bash
   python tests/verify_compile.py
   ```

---

## 📈 Running Experiments

All experiments are driven by configuration files under `configs/`.

To train a specific experiment configuration, run:

```bash
# EXP001: Baseline ResNet50 Classifier
python train.py --experiment_name EXP001_Baseline --model_config configs/EXP001_Baseline.yaml

# EXP006: Proposed TrustOCT Model
python train.py --experiment_name EXP006_TrustOCT --model_config configs/EXP006_TrustOCT.yaml
```

---

## 📊 Evaluation & Diagnostics

After training, you can evaluate the models on unseen external test sets (cross-device verification) or generate visualizations:

1. **Attribution Map Overlay (LayerCAM vs Grad-CAM)**:
   Outputs are saved to `outputs/figures/explainability/`.
2. **Calibration Curves (Reliability Diagrams)**:
   Displays accuracy vs confidence calibration and outputs ECE values.
3. **Confusion Matrix Heatmaps**:
   Generates standard diagnostic classification grids.
