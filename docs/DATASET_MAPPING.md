# Dataset Taxonomy & Cross-Device Mapping Specification

This document defines the authoritative dataset taxonomy, class mapping, and dataset split rules for **TrustOCT** experiments.

---

## 1. Primary Dataset: Kermany OCT2017
- **Purpose**: Training, validation, and internal test baseline.
- **Reference**: Kermany et al., 2018 (Cell).
- **Scanner Brand**: Heidelberg Spectralis (SD-OCT).
- **Taxonomy (4 Classes)**:
  1. `CNV` (Choroidal Neovascularization)
  2. `DME` (Diabetic Macular Edema)
  3. `DRUSEN` (Multiple Drusen / Early AMD)
  4. `NORMAL` (Healthy Retina)
- **Official Split Strategy**:
  - We use the **Official Split** provided by the authors to ensure direct reproducibility and baseline benchmarking with peer-reviewed literature:
    - Train: `datasets/raw/Kermany/train`
    - Val: `datasets/raw/Kermany/val` (or 20% validation split from Train)
    - Test: `datasets/raw/Kermany/test`

---

## 2. External Generalization Target: OCTDL
- **Purpose**: Unseen scanner external testing (Zero-shot generalization).
- **Reference**: Kim et al., 2024 (Scientific Data).
- **Scanner Brand**: Zeiss Cirrus HD-OCT.
- **Taxonomy (7 Classes)**: `AMD`, `DME`, `ERM`, `RAO`, `RVO`, `VID`, `Normal`.
- **Mapping Strategy**:
  Since our model is trained on 4 classes, we map the overlapping classes of OCTDL and exclude the rest:

| OCTDL Class | mapped TrustOCT Class | Inclusion Status | Rationale |
| :--- | :--- | :--- | :--- |
| **AMD** | `DRUSEN` (or `CNV`) | ⚠️ Conditional / Grouped | Advanced/dry AMD corresponds to Drusen/CNV. |
| **DME** | `DME` | ✅ Included | Perfect 1:1 clinical mapping. |
| **Normal** | `NORMAL` | ✅ Included | Perfect 1:1 clinical mapping. |
| **ERM** | (None) | ❌ Excluded | Epiretinal Membrane is not in Kermany. |
| **RAO** | (None) | ❌ Excluded | Retinal Artery Occlusion is not in Kermany. |
| **RVO** | (None) | ❌ Excluded | Retinal Vein Occlusion is not in Kermany. |
| **VID** | (None) | ❌ Excluded | Vitreous Debris is not in Kermany. |

---

## 3. External Generalization Target: NEH-UT
- **Purpose**: Multi-vendor validation (Topcon / Zeiss Cirrus scanner).
- **Taxonomy mapping**: Matches clinical definitions for `DME`, `DRUSEN`, and `NORMAL` wherever possible, ignoring non-overlapping macular pathologies (e.g. CSR, MH).

---

## 4. Dataset Inclusion & Exclusion Rationale
To preserve the mathematical integrity of the classification head without dynamic resizing of weights:
- Unseen target domains are evaluated using the overlapping subset of classes.
- Out-of-Distribution (OOD) metrics treat excluded classes (e.g. ERM, CSR, MH) as negative/OOD cases to verify if our **Evidential Deep Learning** head correctly flags them with high uncertainty.
