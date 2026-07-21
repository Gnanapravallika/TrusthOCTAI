# TrustOCT Framework TODO List

## Phase 1: Setup & Initialization (Active)
- [x] Propose V3.0 codebase architecture
- [x] Create project documents (`PROJECT_SPECIFICATION.md`, `RESEARCH_PLAN.md`, `IMPLEMENTATION_PLAN.md`, `EXPERIMENT_PLAN.md`, `TODO.md`)
- [ ] Create repository skeleton (empty modules and directory layouts)
- [ ] Write `requirements.txt` and configs templates

## Phase 2: Pipeline Development
- [ ] Set up data preprocessing (Bilateral filter, CLAHE) in `src/preprocessing/`
- [ ] Set up dataset loaders in `src/datasets/`
- [ ] Download and verify datasets (Kermany, OCTDL, NEH-UT)
- [ ] Implement backbone modules and head classes in `src/models/`
- [ ] Implement builder registry in `src/registry/`
- [ ] Write tests and verify forward-pass pipeline in `tests/test_pipeline.py`

## Phase 3: Model Training & Baselines
- [ ] Write evidential loss function in `src/losses/`
- [ ] Write training and trainer loop in `src/train/`
- [ ] Execute Baseline training (EXP001)
- [ ] Execute Ablation studies (EXP002 - EXP006)
- [ ] Execute Proposed TrustOCT training (EXP007)

## Phase 4: Verification & Comparison
- [ ] Compute ECE, calibration, and selective prediction statistics
- [ ] Produce plots and reliability diagrams in `paper/figures/`
- [ ] Format classification tables in `paper/tables/`
- [ ] Compile explainability outputs (LayerCAM vs Grad-CAM)
