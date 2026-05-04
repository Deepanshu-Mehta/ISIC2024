# ISIC 2024 — Skin Cancer Detection

Binary classification of malignant vs benign skin lesions from the [ISIC 2024 SLICE-3D dataset](https://www.kaggle.com/competitions/isic-2024-challenge). Three-phase ensemble: tabular GBDT (Phase 1) + image deep learning with cross-modal tabular conditioning (Phase 2) + stacking meta-learner (Phase 3).

**Final pAUC = 0.1745** (partial AUC above 80% TPR)

## Live Demo

**[https://huggingface.co/spaces/Deepanshu027/isic2024-demo](https://huggingface.co/spaces/Deepanshu027/isic2024-demo)**

Gallery of 8 curated cases (3 benign, 3 malignant, 2 edge cases) with OOF predictions from all three pipeline phases. No GPU required at runtime.

### Run locally (Docker)

```bash
docker build -t isic2024-demo .
docker run -p 7860:7860 isic2024-demo
# Open http://localhost:7860
```

## Architecture

```
Phase 1: Tabular GBDT Ensemble (pAUC = 0.1672)
    Raw CSV (401K x 55) -> Preprocessor -> Feature Engineering (88 features)
    -> LightGBM + XGBoost + CatBoost (5-fold, 5-seed avg) -> RankEnsemble -> Isotonic calibration

Phase 2: Image Deep Learning (best pAUC = 0.1588)
    HDF5 images -> EfficientNetV2-S / EVA02 / ConvNeXtV2 / SwinV2-B (timm)
    -> Focal loss + AdamW + OneCycleLR -> TTA x8 (D4 transforms)
    -> SwinV2-B + Tabular conditioning (20-feature MLP fusion)

Phase 3: Stacking Meta-Learner (pAUC = 0.1745)
    5 rank-normalised OOF sources -> LogReg stacker (class_weight='balanced')
```

## Dataset

- **401,059** lesion records from 1,042 patients (~385 lesions/patient)
- **393 malignant** (0.098% positive rate — extreme class imbalance, ~1020:1 ratio)
- Data not included — download from [Kaggle](https://www.kaggle.com/competitions/isic-2024-challenge/data)

## Why Metadata-Only Works

The `tbp_lv_*` columns are machine-extracted measurements from Canfield's Vectra Total Body Photography system, numerically encoding the ABCDE dermatology criteria:

| Criterion | Columns |
|-----------|---------|
| **A**symmetry | `tbp_lv_symm_2axis`, `tbp_lv_eccentricity` |
| **B**order | `tbp_lv_norm_border`, `tbp_lv_perimeterMM` |
| **C**olor | `tbp_lv_deltaLBnorm`, `tbp_lv_color_std_mean`, `tbp_lv_deltaA/B/L` |
| **D**iameter | `clin_size_long_diam_mm`, `tbp_lv_areaMM2` |
| **E**volution | `tbp_lv_nevi_confidence`, `tbp_lv_dnn_lesion_confidence` |

## Phase 1: Tabular GBDT Ensemble

| Component | Choice | Reason |
|-----------|--------|--------|
| CV | `StratifiedGroupKFold(n=5)` | Patient IDs never leak across folds |
| Models | LightGBM, XGBoost, CatBoost | GBDT ensemble for tabular data |
| Baseline | SVM + Platt scaling | Probabilistic model baseline |
| Ensemble | Rank-average (GBDT only) | Robust to probability scale differences |
| Stability | Seed averaging (5 seeds) | Reduces variance from random initialisation |
| Calibration | Isotonic regression on OOF | Improves ECE/Brier; does not affect pAUC |
| Metric | pAUC above 80% TPR | Exact Kaggle competition implementation |

### Feature Pipeline

```
Raw CSV (401,059 x 55)
    -> Preprocessor          -> 44 cols  (drop leakage, encode categoricals, impute)
    -> Feature Engineering   -> 57 cols  (+13: color, shape, interaction, location)
    -> Ugly Duckling         -> 270 cols (+213: patient-wise z-scores + ECDF at 3 group levels)
    -> Feature Selection     -> 88 cols  (correlation threshold 0.90, quasi-constant filter)
```

**Ugly duckling groups** (inspired by 2nd-place ISIC 2024 solution):
1. `[patient_id]` — weird vs patient's other moles?
2. `[patient_id, anatom_site_general]` — weird for this body region?
3. `[patient_id, tbp_lv_location_simple]` — weird for this exact TBP location?

### Leakage Notes

| Column | Status | Reason |
|--------|--------|--------|
| `mel_thick_mm`, `mel_mitotic_index` | Dropped | Only populated for confirmed malignant cases |
| `iddx_full`, `iddx_1`–`iddx_5` | Dropped | Diagnosis labels — direct leakage |
| `lesion_id` | Converted to `has_lesion_id`, then dropped | All 393 malignant cases have `lesion_id`; test set lacks the column |
| `image_type`, `copyright_license` | Dropped | Zero variance / administrative |

### Phase 1 Results (5-Fold CV, seed avg x 5)

| Model | pAUC (mean +/- std) | AUC | Notes |
|-------|---------------------|-----|-------|
| LightGBM | 0.1400 +/- 0.0052 | 0.920 | `min_sum_hessian_in_leaf=10` |
| XGBoost | **0.1703 +/- 0.0074** | 0.964 | Best single model; `max_delta_step=1` critical |
| CatBoost | 0.1667 +/- 0.0064 | 0.960 | `auto_class_weights=SqrtBalanced` |
| SVM baseline | 0.1167 +/- 0.0245 | 0.885 | 20K subsample; high variance expected |
| **GBDT Ensemble** | **0.1653** | 0.956 | Rank-average of LGBM + XGB + CatBoost |
| **Ensemble + Calibration** | **0.1672** | 0.954 | Isotonic regression on OOF |

> Metric: partial AUC above 80% TPR, range [0.0, 0.2]. Uses label-flip + prediction negation — NOT sklearn's `roc_auc_score(max_fpr=0.2)` which applies McClish normalisation.

---

## Phase 2: Image Deep Learning

Fine-tuned pretrained backbones (via [timm](https://github.com/huggingface/pytorch-image-models)) on 401K lesion images stored as JPEG bytes in HDF5, using PyTorch Lightning on Lightning.ai GPUs.

| Component | Choice | Reason |
|-----------|--------|--------|
| Framework | PyTorch Lightning + timm | Pretrained backbones, reproducible training |
| CV | Same `StratifiedGroupKFold(n=5)` as Phase 1 | OOF predictions aligned by `isic_id` for stacking |
| Loss | Focal loss (gamma=2, alpha=0.25) | Handles extreme 1020:1 class imbalance |
| Sampling | WeightedRandomSampler (neg:pos=50:1) | Balanced mini-batches |
| Optimizer | AdamW + OneCycleLR | Differential LRs: backbone (5e-5) vs head (5e-4) |
| TTA | 8 D4 transforms (4 rotations x 2 flips) | Averaged sigmoid probabilities |

### Image Model Results (5-Fold CV, TTA)

| Model | Params | Image Size | OOF pAUC | OOF AUC | GPU |
|-------|--------|------------|----------|---------|-----|
| EfficientNetV2-S | 20M | 224 | 0.1399 | 0.9239 | T4 |
| EVA02-Small | 22M | 336 | 0.1403 | 0.9220 | T4 |
| ConvNeXtV2-B | 88M | 224 | 0.1464 | 0.9286 | L40S |
| SwinV2-B | 87M | 256 | 0.1549 | 0.9388 | L40S |
| **SwinV2-B + Tabular** | **87M** | **256** | **0.1588** | — | **H100** |

**Cross-modal fusion**: The tabular-conditioned model fuses 20 clinical features through a two-layer MLP (Linear(128)->LayerNorm->GELU->Dropout->Linear(64)->LayerNorm->GELU) concatenated with the backbone features before the classification head.

---

## Phase 3: Stacking Meta-Learner

Stacks Phase 1 tabular + Phase 2 image OOF predictions (5 rank-transformed features) through multiple meta-learners. Same CV folds ensure leak-free alignment.

```
Phase 1 OOF (tabular, pAUC=0.1653)      --+
Phase 2 OOF (EfficientNetV2-S, 0.1399)    |
Phase 2 OOF (EVA02-Small, 0.1403)         +-->  Rank Transform -> LogReg -> pAUC = 0.1745
Phase 2 OOF (ConvNeXtV2-B, 0.1464)        |
Phase 2 OOF (SwinV2-B, 0.1549)          --+
```

SwinV2-B + Tabular (pAUC=0.1588) excluded from stacker: highly correlated with vanilla SwinV2-B (same backbone), reducing ensemble diversity. Stacking rewards architectural diversity over individual pAUC.

### Stacking Results

| Method | pAUC | Notes |
|--------|------|-------|
| Rank ensemble (equal weights) | 0.1708 | Zero overfitting risk |
| Rank ensemble (pAUC-weighted) | 0.1715 | Weights by individual pAUC |
| **LogReg stacker** | **0.1745** | `class_weight='balanced'`, 5-fold CV |
| LightGBM stacker | 0.1645 | Conservative params, 3-seed avg — slight overfit |

### Ablation Study (leave-one-out, LogReg stacker)

| Source removed | pAUC | Drop |
|----------------|------|------|
| Tabular GBDT | 0.1633 | -0.0112 (-6.4%) |
| EVA-02 | 0.1737 | -0.0008 |
| ConvNeXtV2 | 0.1735 | -0.0010 |
| SwinV2-B | 0.1736 | -0.0009 |
| EfficientNetV2-S | 0.1748 | +0.0003 |
| **Full ensemble** | **0.1745** | — |

Tabular GBDT is the most impactful single source by a large margin (-6.4% when removed).

---

## Setup

```bash
# Phase 1 (tabular, CPU)
conda env create -f environment.yml
conda activate isic2024
pip install -e .

# Phase 2 (image, GPU — Lightning.ai)
conda env create -f environment_phase2.yml
conda activate isic2024-phase2
pip install -e .
```

## Usage

```bash
# Run tests (134 tests: 129 pass, 5 skip if catboost not installed)
python -m pytest tests/ -v --tb=short

# Lint / format
ruff check src/ tests/
ruff format src/ tests/

# Phase 1: Tabular training
python -m isic2024.train --config configs/base.yaml --output-dir outputs

# Phase 2: Image training (per-backbone, run on Lightning.ai with GPU)
python -m isic2024.train_image --config configs/phase2.yaml --output-dir outputs/phase2/efficientnetv2_s --folds 0,1,2,3,4
python -m isic2024.train_image --config configs/phase2_eva02.yaml --output-dir outputs/phase2/eva02 --folds 0,1,2,3,4
python -m isic2024.train_image --config configs/phase2_convnextv2.yaml --output-dir outputs/phase2/convnextv2 --folds 0,1,2,3,4
python -m isic2024.train_image --config configs/phase2_swin.yaml --output-dir outputs/phase2/swin --folds 0,1,2,3,4

# Phase 3: Stacking meta-learner
python -m isic2024.train_stacking --output-dir outputs/phase3

# Gradio demo (local)
python app/app.py
```

## Training Outputs

| Phase | Directory | Key Files |
|-------|-----------|-----------|
| Phase 1 | `outputs/` | `oof_predictions.csv`, `cv_results.json`, `feature_names.json`, `preprocessor.pkl`, `selector.pkl` |
| Phase 2 | `outputs/phase2/<backbone>/` | `oof_image_predictions.csv`, `fold_assignments.csv`, `fold_*/checkpoints/` |
| Phase 3 | `outputs/phase3/` | `oof_stacking_predictions.csv`, `cv_results.json` |

## Project Structure

```
src/isic2024/
├── config.py              # Phase 1 config dataclasses + YAML loading
├── config_phase2.py       # Phase 2 config dataclasses
├── train.py               # Phase 1: load -> features -> CV -> ensemble -> save
├── train_image.py         # Phase 2: image training with Lightning + TTA
├── train_stacking.py      # Phase 3: stacking meta-learner (tabular + image OOFs)
├── data/
│   ├── loader.py          # CSV loading + column validation
│   ├── preprocess.py      # Imputation, encoding, leakage removal
│   ├── image_dataset.py   # HDF5 image dataset + Lightning DataModule
│   └── augmentation.py    # Train/val/TTA augmentation pipelines
├── features/
│   ├── engineering.py     # Color, shape, interaction features (ABCDE)
│   ├── ugly_duckling.py   # Patient-wise z-scores + ECDF at 3 group levels
│   ├── selection.py       # Correlation + variance + quasi-constant filtering
│   └── pipeline.py        # Orchestrates preprocess -> engineer -> UD -> select
├── models/
│   ├── gbdt.py            # LightGBM / XGBoost / CatBoost wrappers
│   ├── svm_baseline.py    # SVM + StandardScaler + Platt scaling
│   ├── ensemble.py        # Rank-averaging ensemble
│   ├── calibration.py     # Isotonic / Platt / Temperature calibration
│   ├── image_module.py    # Lightning module: timm backbone + head
│   └── losses.py          # Focal loss for image models
└── evaluation/
    ├── metrics.py         # pAUC (exact competition impl), ECE, Brier
    └── plots.py           # ROC curves, score distributions, reliability diagrams
app/
├── app.py                 # Gradio web demo (gallery + predictions)
├── requirements.txt       # Runtime dependencies (gradio only, no GPU)
└── gallery/               # Precomputed cases (images + JSON)
configs/
├── base.yaml              # Phase 1 hyperparameters
├── phase2.yaml            # EfficientNetV2-S
├── phase2_eva02.yaml      # EVA02-Small (336px)
├── phase2_convnextv2.yaml # ConvNeXtV2-B (224px)
├── phase2_swin.yaml       # SwinV2-B (256px)
└── phase2_swin_tabular.yaml  # SwinV2-B + tabular conditioning
tests/                     # 134 tests (pytest)
Dockerfile                 # Containerised Gradio demo
.github/workflows/         # CI: lint + test on push/PR
```
