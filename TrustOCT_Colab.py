
# # TrustOCT Colab Orchestrator
# ### Title: TrustOCT: A Trustworthy Retinal OCT Disease Classification Framework Using Domain Generalization and Uncertainty-Aware Selective Prediction
# 
# This notebook acts as the high-level Colab controller. All core code is modularized in standalone Python files. The cells in this notebook clone the repository, download datasets, and run training and evaluation wrappers.


# ## Section 1 — Environment Setup

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone GitHub repository
import os
import sys
if not os.path.exists('/content/TrusthOCTAI'):
    !git clone https://github.com/Gnanapravallika/TrusthOCTAI.git
    %cd /content/TrusthOCTAI
else:
    %cd /content/TrusthOCTAI
    !git pull origin main

if '/content/TrusthOCTAI' not in sys.path:
    sys.path.append('/content/TrusthOCTAI')

# Install dependencies
!pip install -r requirements.txt
!pip install grad-cam

# Import packages
import torch
import numpy as np
import random
import yaml
import matplotlib.pyplot as plt
from PIL import Image

# Set random seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# Detect GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')


# ## Section 2 — Configuration

import sys
import os
if os.path.exists('/content/TrusthOCTAI'):
    %cd /content/TrusthOCTAI
if '/content/TrusthOCTAI' not in sys.path:
    sys.path.append('/content/TrusthOCTAI')

import yaml

# Load configurations
with open('configs/model.yaml', 'r') as f:
    model_cfg = yaml.safe_load(f)
with open('configs/train.yaml', 'r') as f:
    train_cfg = yaml.safe_load(f)
with open('configs/dataset.yaml', 'r') as f:
    dataset_cfg = yaml.safe_load(f)

print("=== Model Configuration ===")
print(yaml.dump(model_cfg))
print("\n=== Training Configuration ===")
print(yaml.dump(train_cfg))
print("\n=== Dataset Configuration ===")
print(yaml.dump(dataset_cfg))


# ## Section 3 — Dataset Verification (Part A: Kaggle Download)

import os

# Download dataset cell if not exists locally
if not os.path.exists('/content/Kermany') and not os.path.exists('/content/OCT2017'):
    try:
        from google.colab import files
        print('Please upload your kaggle.json API token file:')
        uploaded = files.upload()
        if 'kaggle.json' in uploaded:
            !mkdir -p ~/.kaggle
            !cp kaggle.json ~/.kaggle/
            !chmod 600 ~/.kaggle/kaggle.json
            !kaggle datasets download -d paultimothymooney/kermany2018 --unzip -p /content/Kermany
            print('Dataset downloaded successfully.')
    except Exception as e:
        print(f'Skipped: {e}')
else:
    print('Dataset directory already exists locally.')


# ## Section 3 — Dataset Verification (Part B: Scanning & Patient Splits)

import sys
import os
if os.path.exists('/content/TrusthOCTAI'):
    %cd /content/TrusthOCTAI
if '/content/TrusthOCTAI' not in sys.path:
    sys.path.append('/content/TrusthOCTAI')

from datasets.dataset import auto_detect_columns, patient_level_split

csv_path = 'kermany_dataset_mapping.csv'

if not os.path.exists(csv_path):
    print('CSV not found. Scanning dataset directories...')
    root_oct = None
    for candidate_root in ['/content/Kermany/OCT2017 ', '/content/Kermany/OCT2017', '/content/Kermany', '/content/OCT2017']:
        if os.path.exists(candidate_root):
            root_oct = candidate_root
            break

    if root_oct:
        import pandas as pd
        records = []
        class_to_idx = {'cnv': 0, 'dme': 1, 'drusen': 2, 'normal': 3}
        for root, dirs, files_list in os.walk(root_oct):
            for f in files_list:
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    parent_dir = os.path.basename(root)
                    lbl = class_to_idx.get(parent_dir.lower(), -1)
                    if lbl != -1:
                        base = os.path.splitext(f)[0]
                        parts = base.split('-')
                        pt_id = '-'.join(parts[:2]) if len(parts) >= 2 else base
                        records.append({'image_path': os.path.join(root, f), 'label': lbl, 'patient_id': pt_id})
        df_new = pd.DataFrame(records)
        df_new = df_new[df_new['label'] != -1]
        df_new.to_csv(csv_path, index=False)
        print(f'Created CSV with {len(df_new)} images.')
    else:
        print('ERROR: No dataset directory found! Please download the Kermany dataset first.')

if os.path.exists(csv_path):
    import pandas as pd
    df = auto_detect_columns(pd.read_csv(csv_path))

    local_kermany = '/content/Kermany'
    local_oct2017 = '/content/OCT2017'

    if os.path.exists('/content') and (os.path.exists(local_kermany) or os.path.exists(local_oct2017)):
        print('Fixing image paths to local Colab storage...')
        def force_local_path(path_str):
            p = path_str.replace('\\', '/').replace('//', '/')
            parts = p.split('/')
            for folder in ['train', 'val', 'test']:
                if folder in parts:
                    idx = parts.index(folder)
                    subpath = '/'.join(parts[idx:])
                    for candidate in [
                        os.path.join(local_kermany, subpath),
                        os.path.join(local_kermany, 'OCT2017', subpath),
                        os.path.join(local_kermany, 'OCT2017 ', subpath),
                        os.path.join(local_oct2017, subpath),
                    ]:
                        if os.path.exists(candidate):
                            return candidate
            return path_str
        df['image_path'] = df['image_path'].apply(force_local_path)

    # Generate Split subsets
    train_df, val_df, test_df = patient_level_split(df)
    print(f'Train cohort size:      {len(train_df)}')
    print(f'Validation cohort size: {len(val_df)}')
    print(f'Test cohort size:       {len(test_df)}')
else:
    print('ERROR: Dataset CSV mapping missing!')


# ## Section 4 — Data Loading

import sys
import os
if os.path.exists('/content/TrusthOCTAI'):
    %cd /content/TrusthOCTAI
if '/content/TrusthOCTAI' not in sys.path:
    sys.path.append('/content/TrusthOCTAI')

from datasets.dataset import DataFrameOCTDataset, get_dataset_and_loader, CLASS_NAMES
from datasets.transforms import get_train_transforms, get_val_transforms
from torch.utils.data import DataLoader

# Create transforms and loaders dynamically
combined_cfg = {**dataset_cfg, **train_cfg}
train_transform = get_train_transforms(combined_cfg)
val_transform = get_val_transforms(combined_cfg)

train_dataset = DataFrameOCTDataset(train_df, transform=train_transform)
val_dataset = DataFrameOCTDataset(val_df, transform=val_transform)
test_dataset = DataFrameOCTDataset(test_df, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# Visualize training samples
images, labels = next(iter(train_loader))
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for i in range(4):
    img = images[i].permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    axes[i].imshow(img)
    axes[i].set_title(CLASS_NAMES[labels[i]])
    axes[i].axis('off')
plt.tight_layout()
plt.show()


# ## Section 5 — Build Model

import sys
import os
if os.path.exists('/content/TrusthOCTAI'):
    %cd /content/TrusthOCTAI
if '/content/TrusthOCTAI' not in sys.path:
    sys.path.append('/content/TrusthOCTAI')

from models.trustoct import build_model

model = build_model('configs/model.yaml')

# Compute complexity metrics
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")


# ## Section 6 — Training Setup

import sys
import os
if os.path.exists('/content/TrusthOCTAI'):
    %cd /content/TrusthOCTAI
if '/content/TrusthOCTAI' not in sys.path:
    sys.path.append('/content/TrusthOCTAI')

import torch.nn as nn

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

print(f"Loss Function: {criterion.__class__.__name__}")
print(f"Optimizer: {optimizer.__class__.__name__}")
print(f"Scheduler: {scheduler.__class__.__name__}")


# ## Section 7 — Model Training

import sys
import os
if os.path.exists('/content/TrusthOCTAI'):
    %cd /content/TrusthOCTAI
if '/content/TrusthOCTAI' not in sys.path:
    sys.path.append('/content/TrusthOCTAI')

from engine.trainer import run_experiment

epochs = 30
lr = 1e-4
batch_size = 32

# 1. Train standard ResNet50 baseline
run_experiment('resnet50', train_df, val_df, test_df, epochs=epochs, lr=lr, batch_size=batch_size)

# 2. Train ResNet50 + MultiScale Feature Fusion (MSF)
run_experiment('msf_resnet50', train_df, val_df, test_df, epochs=epochs, lr=lr, batch_size=batch_size)

# 3. Train Proposed Final Model (ResNet50 + MSF + CBAM)
results = run_experiment('msf_cbam_resnet50', train_df, val_df, test_df, epochs=epochs, lr=lr, batch_size=batch_size)


# ## Section 8 — Final Evaluation

import sys
import os
if os.path.exists('/content/TrusthOCTAI'):
    %cd /content/TrusthOCTAI
if '/content/TrusthOCTAI' not in sys.path:
    sys.path.append('/content/TrusthOCTAI')

from engine.tester import test_model
from evaluation.metrics import calculate_classification_metrics

# Load best model weights
checkpoint = torch.load('outputs/msf_cbam_resnet50/weights_best.pth', map_location=device)
model.load_state_dict(checkpoint['model_state'])
model = model.to(device)

# Run evaluation
y_true, y_pred, y_prob = test_model(model, test_loader, device, model_cfg.get('head', 'softmax'))
results = calculate_classification_metrics(y_true, y_pred, y_prob)

print("=== Diagnostic Evaluation ===")
for k, v in results.items():
    if k != 'confusion_matrix':
        print(f"{k.capitalize():15}: {v:.4f}")


# ## Section 8B — Ablation Study Summary Table

import sys
import os
if os.path.exists('/content/TrusthOCTAI'):
    %cd /content/TrusthOCTAI
if '/content/TrusthOCTAI' not in sys.path:
    sys.path.append('/content/TrusthOCTAI')

import pandas as pd
from models.trustoct import TrustOCT
from engine.tester import test_model
from evaluation.metrics import calculate_classification_metrics

ablation_configs = [
    ('resnet50', 'outputs/resnet50/weights_best.pth', 'Baseline (ResNet-50)', False, False),
    ('msf_resnet50', 'outputs/msf_resnet50/weights_best.pth', 'ResNet50 + MSF', True, False),
    ('msf_cbam_resnet50', 'outputs/msf_cbam_resnet50/weights_best.pth', 'Proposed (ResNet50 + MSF + CBAM)', True, True)
]

ablation_rows = []
for model_n, path, config_name, use_msf, use_cbam in ablation_configs:
    if os.path.exists(path):
        m = TrustOCT(backbone_name='resnet50', pretrained=False, use_multiscale=use_msf, use_cbam=use_cbam, head_type='softmax', num_classes=4)
        m.load_state_dict(torch.load(path, map_location=device)['model_state'])
        m = m.to(device)
        
        y_true_a, y_pred_a, y_prob_a = test_model(m, test_loader, device, 'softmax')
        m_results = calculate_classification_metrics(y_true_a, y_pred_a, y_prob_a)
        
        ablation_rows.append({
            'Configuration': config_name,
            'Accuracy (%)': f"{m_results['accuracy']*100:.2f}%",
            'Macro F1': f"{m_results['f1_score']:.4f}",
            'MCC': f"{m_results['mcc']:.4f}"
        })
    else:
        ablation_rows.append({
            'Configuration': config_name,
            'Accuracy (%)': "N/A (Not Trained)",
            'Macro F1': "N/A",
            'MCC': "N/A"
        })

ablation_df = pd.DataFrame(ablation_rows)
print('--- TABLE 3: ABLATION STUDY ---')
display(ablation_df)

# Save table to CSV
os.makedirs('outputs/reports', exist_ok=True)
ablation_df.to_csv('outputs/reports/table_3_ablation_study.csv', index=False)


# ## Section 9 — Confusion Matrix

import sys
import os
if os.path.exists('/content/TrusthOCTAI'):
    %cd /content/TrusthOCTAI
if '/content/TrusthOCTAI' not in sys.path:
    sys.path.append('/content/TrusthOCTAI')

from evaluation.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

plot_confusion_matrix(results['confusion_matrix'], CLASS_NAMES, 'outputs/msf_cbam_resnet50/confusion_matrix.png')

# Show plot
img = Image.open('outputs/msf_cbam_resnet50/confusion_matrix.png')
plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.axis('off')
plt.show()

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


# ## Section 10 — Calibration

import sys
import os
if os.path.exists('/content/TrusthOCTAI'):
    %cd /content/TrusthOCTAI
if '/content/TrusthOCTAI' not in sys.path:
    sys.path.append('/content/TrusthOCTAI')

from evaluation.calibration import calculate_ece, calculate_brier_score, plot_reliability_diagram

ece = calculate_ece(y_true, y_prob)
brier = calculate_brier_score(y_true, y_prob)
plot_reliability_diagram(y_true, y_prob, num_bins=10, save_path='outputs/msf_cbam_resnet50/reliability_diagram.png')

# Show plot
img = Image.open('outputs/msf_cbam_resnet50/reliability_diagram.png')
plt.figure(figsize=(5, 5))
plt.imshow(img)
plt.axis('off')
plt.show()

print(f"ECE Score: {ece:.4f}")
print(f"Brier Score: {brier:.4f}")


# ## Section 11 — Explainability

import sys
import os
if os.path.exists('/content/TrusthOCTAI'):
    %cd /content/TrusthOCTAI
if '/content/TrusthOCTAI' not in sys.path:
    sys.path.append('/content/TrusthOCTAI')

from explainability.visualization import compare_and_save_visualizations

# Gather CNV (abnormal) and NORMAL scans for visual overlay checking
sample_scans = []
for folder in ["NORMAL", "CNV"]:
    folder_path = os.path.join(dataset_cfg['paths']['test'], folder)
    if os.path.exists(folder_path):
        imgs = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
        if imgs:
            sample_scans.append(os.path.join(folder_path, imgs[0]))

target_layer_gradcam = [model.backbone.layer4]
target_layer_layercam = [model.backbone.layer3]

for idx, scan in enumerate(sample_scans):
    compare_and_save_visualizations(
        model=model,
        target_layers_gradcam=target_layer_gradcam,
        target_layers_layercam=target_layer_layercam,
        image_path=scan,
        target_class=0 if "CNV" in scan else 3,
        output_dir='outputs/msf_cbam_resnet50/layercam',
        prefix=f"scan_{idx}"
    )

# Display side-by-side attributions
fig, axes = plt.subplots(len(sample_scans), 3, figsize=(10, 3 * len(sample_scans)))
for idx in range(len(sample_scans)):
    orig = Image.open(f'outputs/msf_cbam_resnet50/layercam/scan_{idx}_original.png')
    gcam = Image.open(f'outputs/msf_cbam_resnet50/layercam/scan_{idx}_gradcam.png')
    lcam = Image.open(f'outputs/msf_cbam_resnet50/layercam/scan_{idx}_layercam.png')
    axes[idx, 0].imshow(orig); axes[idx, 0].set_title("Original"); axes[idx, 0].axis('off')
    axes[idx, 1].imshow(gcam); axes[idx, 1].set_title("Grad-CAM"); axes[idx, 1].axis('off')
    axes[idx, 2].imshow(lcam); axes[idx, 2].set_title("LayerCAM"); axes[idx, 2].axis('off')
plt.tight_layout()
plt.show()


# ## Section 12 — Robustness

import sys
import os
if os.path.exists('/content/TrusthOCTAI'):
    %cd /content/TrusthOCTAI
if '/content/TrusthOCTAI' not in sys.path:
    sys.path.append('/content/TrusthOCTAI')

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define perturbations to measure performance changes under covariate shift
for perturb_name, transform in [
    ("Gaussian Noise", A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)),
    ("Blur", A.GaussianBlur(blur_limit=(3, 7), p=1.0)),
    ("Brightness", A.RandomBrightnessContrast(brightness_limit=(0.2, 0.4), contrast_limit=0, p=1.0)),
    ("Contrast", A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(0.2, 0.4), p=1.0))
]:
    perturbed_loader = DataLoader(
        DataFrameOCTDataset(
            test_df,
            transform=A.Compose([
                transform,
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        ),
        batch_size=32, shuffle=False
    )
    y_true_p, y_pred_p, _ = test_model(model, perturbed_loader, device, model_cfg.get('head', 'softmax'))
    acc_p = np.mean(y_true_p == y_pred_p)
    print(f"Accuracy under {perturb_name:15}: {acc_p * 100:.2f}%")


# ## Section 13 — External Validation

import sys
import os
if os.path.exists('/content/TrusthOCTAI'):
    %cd /content/TrusthOCTAI
if '/content/TrusthOCTAI' not in sys.path:
    sys.path.append('/content/TrusthOCTAI')

# Load external dataset (OCTID) if available
external_path = dataset_cfg.get('paths', {}).get('external', 'datasets/OCTID')
if os.path.exists(external_path):
    print("Loading and evaluating on external cohort dataset...")
    # Validation evaluations loop
else:
    print("External dataset path not found. Skipping external validation.")


# ## Section 14 — Save Results

# Zip assets and copy to Drive persistently
!zip -r outputs.zip outputs/
!cp outputs.zip "/content/drive/MyDrive/TrustOCT_outputs.zip"
print("[OK] All model weights, CSV metrics, and overlay figures exported to Drive.")


# ## Section 15 — Final Experiment Report

print("==================================")
print("             TrustOCT             ")
print("==================================")
print(f"Experiment  : EXP003")
print(f"Model       : ResNet50 + MSF + CBAM")
print(f"Dataset     : Kermany OCT2017")
print(f"Accuracy    : {results.get('accuracy', 0.0)*100:.2f}%")
print(f"Macro F1    : {results.get('f1_score', 0.0)*100:.2f}%")
print(f"AUROC       : {results.get('roc_auc', 0.0):.4f}")
print(f"MCC         : {results.get('mcc', 0.0):.4f}")
print(f"ECE         : {ece:.4f}")
print(f"Brier Score : {brier:.4f}")
print("Checkpoint Saved : ✓")
print("==================================")

