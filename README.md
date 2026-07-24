# TrustOCT: A Trustworthy Retinal OCT Classification Framework

TrustOCT is a trustworthy deep learning framework designed for retinal Optical Coherence Tomography (OCT) disease classification. The framework enhances classification accuracy and confidence calibration by integrating Adaptive Multi-Scale Feature Fusion and CBAM attention gating on a ResNet50 backbone.

---

## 🚀 Key Framework Architecture

Our modular framework follows a configuration-driven pipeline allowing full ablation studies:
1. **CNN Backbone**: Extract multi-level representations (ResNet50 intermediate features).
2. **Adaptive Multi-Scale Fusion**: Aligns and fuses deep representations (`Layer3` and `Layer4`) to prevent resolution loss.
3. **CBAM Attention**: Employs spatial and channel attention to focus on pathology regions (e.g., fluid pockets, drusen).
4. **Softmax Head**: Standard linear projection layer mapping to logits, regularized using dropout.

---

## 📁 Repository Structure

```
TrusthOCTAI/
│
├── configs/                # Experiment YAML configuration files
│   ├── dataset.yaml        # Dataset splits and loader configs
│   ├── augmentation.yaml   # CLAHE, Bilateral filtering, and flip settings
│   ├── train.yaml          # Hyperparameters and optimization setting
│   ├── EXP001_Baseline.yaml# Baseline configuration
│   ├── EXP002_MultiScale.yaml # ResNet50 + MSF configuration
│   └── EXP003_CBAM.yaml    # Proposed final model configuration
│
├── datasets/               # Preprocessing pipelines and loaders
│   ├── dataset.py          # DataLoaders & patient-level splits
│   ├── transforms.py       # CLAHE & Bilateral filter augmentations
│   └── utils.py            # Pre-training folder audits & statistics
│
├── models/                 # Model architectures
│   ├── resnet50.py         # ResNet50 dual-layer backbone
│   ├── msf.py              # Multi-Scale Feature Fusion module
│   ├── cbam.py             # CBAM attention block module
│   └── trustoct.py         # Unified framework assembler & weight initializer
│
├── engine/                 # Training and validation loops
│   ├── losses.py           # Cross-entropy loss function
│   ├── trainer.py          # fit/validate epoch optimization loops
│   ├── validator.py        # Validation split evaluation helpers
│   └── tester.py           # Test set prediction loops
│
├── evaluation/             # Clinical metrics & calibration loops
│   ├── metrics.py          # Macro diagnostic scores & Confusion Matrices
│   ├── calibration.py      # ECE/Brier scoring & Reliability Diagrams
│   ├── robustness.py       # Covariate perturbation checks (noise, blur)
│   └── report.py           # External validation metric reporting
│
├── explainability/         # Attribution map visualizers
│   ├── layercam.py         # LayerCAM and Grad-CAM map extractors
│   └── visualization.py    # Side-by-side overlay heatmap generators
│
├── outputs/                # Folder for reports, logs, and checkpoints
│
├── tests/                  # Integration tests
│   ├── verify_compile.py   # Compilation validation check
│   └── smoke_test.py       # End-to-end sandbox pipeline check
│
├── TrustOCT_Colab.ipynb    # Master Colab Orchestrator
├── main.py                 # Master orchestrator script
├── train.py                # Master training execution wrapper
├── test.py                 # Evaluation execution wrapper
├── inference.py            # Single scan diagnostic utility
├── requirements.txt        # Package dependencies list
└── .gitignore
```

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Gnanapravallika/TrusthOCTAI.git
   cd TrusthOCTAI
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

To train a specific experiment configuration from the ablation registry, run:

```bash
# EXP001: Baseline ResNet50 Classifier
python train.py --experiment_name EXP001_Baseline --model_config configs/EXP001_Baseline.yaml

# EXP003: Proposed final reference model (ResNet50 + MSF + CBAM)
python train.py --experiment_name EXP003_CBAM --model_config configs/EXP003_CBAM.yaml
```

---

## 📊 Evaluation & Diagnostics

After training, you can evaluate the models on unseen external test sets (cross-device verification) or generate visualizations:

1. **Attribution Map Overlay (LayerCAM vs Grad-CAM)**:
   Outputs are saved to `outputs/{experiment_name}/layercam/`.
2. **Calibration Curves (Reliability Diagrams)**:
   Displays accuracy vs confidence calibration and outputs ECE values under `outputs/{experiment_name}/reliability_diagram.png`.
3. **Confusion Matrix Heatmaps**:
   Generates standard diagnostic classification grids under `outputs/{experiment_name}/confusion_matrix.png`.
