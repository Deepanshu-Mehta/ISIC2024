# ISIC 2024 — Skin Cancer Detection

Binary classification of malignant vs benign skin lesions from the [ISIC 2024 SLICE-3D dataset](https://www.kaggle.com/competitions/isic-2024-challenge). Two-phase approach: tabular GBDT ensemble (Phase 1) + image deep learning (Phase 2) + stacking meta-learner (Phase 3).

## Why Metadata-Only Works

The `tbp_lv_*` columns are machine-extracted measurements from Canfield's Vectra Total Body Photography system, numerically encoding the ABCDE dermatology criteria:

| Criterion | Columns |
|-----------|---------|
| **A**symmetry | `tbp_lv_symm_2axis`, `tbp_lv_eccentricity` |
| **B**order | `tbp_lv_norm_border`, `tbp_lv_perimeterMM` |
| **C**olor | `tbp_lv_deltaLBnorm`, `tbp_lv_color_std_mean`, `tbp_lv_deltaA/B/L` |
| **D**iameter | `clin_size_long_diam_mm`, `tbp_lv_areaMM2` |
| **E**volution | `tbp_lv_nevi_confidence`, `tbp_lv_dnn_lesion_confidence` |

## Dataset

- **401,059** lesion records from 1,042 patients (~385 lesions/patient)
- **393 malignant** (0.098% positive rate — extreme class imbalance)
- Data not included — download from [Kaggle](https://www.kaggle.com/competitions/isic-2024-challenge/data)

## Approach

| Component | Choice | Reason |
|-----------|--------|--------|
| CV | `StratifiedGroupKFold(n=5)` | Patient IDs never leak across folds |
| Models | LightGBM, XGBoost, CatBoost | GBDT ensemble for tabular data |
| Baseline | SVM + Platt scaling | Course requirement: probabilistic model |
| Ensemble | Rank-average (GBDT only) | Robust to probability scale differences |
| Stability | Seed averaging (5 seeds) | Reduces variance from random initialisation |
| Calibration | Isotonic regression on OOF | Improves ECE/Brier; does not affect pAUC |
| Metric | pAUC above 80% TPR | Exact Kaggle competition implementation |

## Feature Pipeline

```
Raw CSV (401,059 × 55)
    ↓ Preprocessor          → 44 cols  (drop leakage, encode categoricals, impute)
    ↓ Feature Engineering   → 57 cols  (+13: color, shape, interaction, location)
    ↓ Ugly Duckling         → 270 cols (+213: patient-wise z-scores + ECDF at 3 group levels)
    ↓ Feature Selection     → 88 cols  (correlation threshold 0.90, quasi-constant filter)
```

**Ugly duckling groups** (from 2nd-place ISIC 2024 solution):
1. `[patient_id]` — weird vs patient's other moles?
2. `[patient_id, anatom_site_general]` — weird for this body region?
3. `[patient_id, tbp_lv_location_simple]` — weird for this exact TBP location?

### Leakage Notes

| Column | Status | Reason |
|--------|--------|--------|
| `mel_thick_mm`, `mel_mitotic_index` | Dropped | Only populated for confirmed malignant cases |
| `iddx_full`, `iddx_1`–`iddx_5` | Dropped | Diagnosis labels — direct leakage |
| `lesion_id` | Converted → `has_lesion_id`, then **excluded from features** | All 393 malignant cases have a `lesion_id`; but test set has no `lesion_id` column — model would predict near-zero for all test rows |
| `image_type`, `copyright_license` | Dropped | Zero variance / administrative |

## Benchmark Results (5-Fold CV, seed avg × 5)

| Model | pAUC (mean ± std) | AUC | Notes |
|-------|-------------------|-----|-------|
| LightGBM | 0.1400 ± 0.0052 | 0.920 | `min_sum_hessian_in_leaf=10` fixes imbalance handling |
| XGBoost | **0.1703 ± 0.0074** | 0.964 | Best single model; `max_delta_step=1` critical |
| CatBoost | 0.1667 ± 0.0064 | 0.960 | `auto_class_weights=SqrtBalanced` |
| SVM baseline | 0.1167 ± 0.0245 | 0.885 | 20K subsample; high variance expected |
| **GBDT Ensemble** | **0.1653** | 0.956 | Rank-average of LGBM + XGB + CatBoost |
| **Ensemble + Calibration** | **0.1672** | 0.954 | Isotonic regression on OOF |

**Phase 1 target (pAUC ≥ 0.16): ✅ Achieved at 0.1672**

> Metric: partial AUC above 80% TPR, range [0.0, 0.2]. Uses label-flip + prediction negation — NOT sklearn's `roc_auc_score(max_fpr=0.2)` which applies McClish normalisation.

---

## Phase 2: Image Deep Learning

Fine-tuned pretrained backbones (via [timm](https://github.com/huggingface/pytorch-image-models)) on 401K lesion images stored as JPEG bytes in HDF5, using PyTorch Lightning on Lightning.ai GPUs.

### Approach

| Component | Choice | Reason |
|-----------|--------|--------|
| Framework | PyTorch Lightning + timm | Pretrained backbones, reproducible training |
| CV | Same `StratifiedGroupKFold(n=5)` as Phase 1 | OOF predictions aligned by `isic_id` for stacking |
| Loss | Focal loss (γ=2, α=0.25) | Handles extreme 1020:1 class imbalance |
| Sampling | WeightedRandomSampler (neg:pos=50:1) | Balanced mini-batches |
| Optimizer | AdamW + OneCycleLR | Differential LRs: backbone (5e-5) vs head (5e-4) |
| TTA | 8 D4 transforms (4 rotations × 2 flips) | Averaged sigmoid probabilities |
| Augmentation | H/V flip, rotate90, transpose, color jitter, Gaussian blur, coarse dropout | D4 symmetries + color robustness |

### Image Model Results (5-Fold CV, TTA)

| Model | Params | Image Size | OOF pAUC | OOF AUC | GPU | Time |
|-------|--------|------------|----------|---------|-----|------|
| EfficientNetV2-S | 20M | 224 | 0.1399 | 0.9239 | T4 | ~4hrs |
| EVA02-Small | 22M | 336 | 0.1403 | 0.9220 | T4 | ~10hrs |
| **ConvNeXtV2-B** | **88M** | **224** | **0.1464** | **0.9286** | **L40S** | **~4hrs** |
| SwinV2-B | 88M | 256 | — | — | — | Pending |

**Best image model: ConvNeXtV2-B at pAUC=0.1464**

### Per-Fold Breakdown

| Fold | EfficientNetV2-S | EVA02-Small | ConvNeXtV2-B |
|------|-----------------|-------------|-------------|
| 0 | 0.1383 | 0.1502 | 0.1519 |
| 1 | 0.1528 | 0.1557 | 0.1513 |
| 2 | 0.1551 | 0.1704 | 0.1705 |
| 3 | 0.1106 | 0.1321 | 0.1334 |
| 4 | 0.1671 | 0.1669 | 0.1668 |

> Fold 3 is consistently the hardest across all models and phases, suggesting genuinely difficult cases in that patient split.

### Phase 2 Training

```bash
# Setup (Lightning.ai GPU instance)
conda env create -f environment_phase2.yml
conda activate isic2024-phase2
pip install -e .

# Train per backbone (each saves OOF CSV to its own output dir)
python -m isic2024.train_image --config configs/phase2.yaml --output-dir outputs/phase2/efficientnetv2_s --folds 0,1,2,3,4
python -m isic2024.train_image --config configs/phase2_eva02.yaml --output-dir outputs/phase2/eva02 --folds 0,1,2,3,4
python -m isic2024.train_image --config configs/phase2_convnextv2.yaml --output-dir outputs/phase2/convnextv2 --folds 0,1,2,3,4
python -m isic2024.train_image --config configs/phase2_swin.yaml --output-dir outputs/phase2/swin --folds 0,1,2,3,4
```

### Phase 2 Outputs

Each backbone saves to `outputs/phase2/<backbone>/` (gitignored):

| File | Contents |
|------|----------|
| `oof_image_predictions.csv` | Per-row OOF predictions (isic_id, target, image_pred) |
| `fold_assignments.csv` | Fold index per sample |
| `fold_*/checkpoints/` | Model checkpoints (best + last) |

---

## Phase 3: Stacking (Planned)

Stack tabular OOF + image OOF(s) through a GBDT meta-learner. Same CV folds ensure leak-free alignment.

```
Phase 1 OOF (tabular, pAUC=0.1672)  ─┐
Phase 2 OOF (EfficientNetV2-S)       ├──→ GBDT Meta-Learner → Final pAUC
Phase 2 OOF (EVA02-Small)            │
Phase 2 OOF (ConvNeXtV2-B)          ─┘
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate isic2024
pip install -e .
```

## Usage

```bash
# Run tests (128 tests)
python -m pytest tests/ -v --tb=short

# Lint / format
ruff check src/ tests/
ruff format src/ tests/

# Launch notebooks
jupyter lab

# Train pipeline (saves to outputs/)
python -m src.isic2024.train --config configs/base.yaml
```

## Training Outputs

After running `train.py`, the following are saved to `outputs/` (gitignored):

| File | Contents |
|------|----------|
| `oof_predictions.csv` | Per-row OOF predictions for all models + ensemble |
| `cv_results.json` | Per-fold and mean pAUC / AUC / Brier / ECE |
| `feature_names.json` | 88 selected features (use for test-set inference) |
| `preprocessor.pkl` | Fitted preprocessor (imputation values, label encoders) |
| `selector.pkl` | Fitted feature selector (selected column list) |

## Project Structure

```
src/isic2024/
├── config.py              # Phase 1 config dataclasses + YAML loading
├── config_phase2.py       # Phase 2 config dataclasses
├── train.py               # Phase 1: load → features → CV → ensemble → save
├── train_image.py         # Phase 2: image training with Lightning + TTA
├── data/
│   ├── loader.py          # CSV loading + column validation
│   ├── preprocess.py      # Imputation, encoding, leakage removal
│   ├── image_dataset.py   # HDF5 image dataset + Lightning DataModule
│   └── augmentation.py    # Train/val/TTA augmentation pipelines
├── features/
│   ├── engineering.py     # Color, shape, interaction features (ABCDE)
│   ├── ugly_duckling.py   # Patient-wise z-scores + ECDF at 3 group levels
│   ├── selection.py       # Correlation + variance + quasi-constant filtering
│   └── pipeline.py        # Orchestrates preprocess → engineer → UD → select
├── models/
│   ├── gbdt.py            # LightGBM / XGBoost / CatBoost wrappers (sklearn API)
│   ├── svm_baseline.py    # SVM + StandardScaler + Platt scaling
│   ├── ensemble.py        # Rank-averaging ensemble
│   ├── calibration.py     # Isotonic (primary) / Platt / Temperature calibration
│   ├── image_module.py    # Lightning module: timm backbone + head
│   └── losses.py          # Focal loss for image models
├── evaluation/
│   ├── metrics.py         # pAUC (exact competition impl), ECE, Brier
│   └── plots.py           # ROC curves, score distributions, reliability diagrams, feature importance
configs/
├── base.yaml              # Phase 1 hyperparameters
├── phase2.yaml            # Phase 2 base config (EfficientNetV2-S)
├── phase2_eva02.yaml      # EVA02-Small (336px)
├── phase2_convnextv2.yaml # ConvNeXtV2-B (224px)
└── phase2_swin.yaml       # SwinV2-B (256px)
tests/                      # 128 tests (pytest)
notebooks/
├── 01_eda.ipynb            # Leakage audit, AI scores, ugly duckling motivation
├── 02_feature_exploration.ipynb  # Pipeline validation, correlation threshold sensitivity
├── 03_model_comparison.ipynb     # pAUC bar chart, ROC curves, score distributions, feature importance
└── 04_calibration_analysis.ipynb # Reliability diagrams, ECE/Brier table, pAUC preservation
```
