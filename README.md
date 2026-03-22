# ISIC 2024 — Phase 1: Skin Cancer Detection (Tabular ML)

Binary classification of malignant vs benign skin lesions using **metadata features only** from the [ISIC 2024 SLICE-3D dataset](https://www.kaggle.com/competitions/isic-2024-challenge). No images in Phase 1.

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

## Setup

```bash
conda env create -f environment.yml
conda activate isic2024
pip install -e .
```

## Usage

```bash
# Run tests (124 tests)
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
├── config.py              # Config dataclasses + YAML loading
├── data/
│   ├── loader.py          # CSV loading + column validation
│   └── preprocess.py      # Imputation, encoding, leakage removal
├── features/
│   ├── engineering.py     # Color, shape, interaction features (ABCDE)
│   ├── ugly_duckling.py   # Patient-wise z-scores + ECDF at 3 group levels
│   ├── selection.py       # Correlation + variance + quasi-constant filtering
│   └── pipeline.py        # Orchestrates preprocess → engineer → UD → select
├── models/
│   ├── gbdt.py            # LightGBM / XGBoost / CatBoost wrappers (sklearn API)
│   ├── svm_baseline.py    # SVM + StandardScaler + Platt scaling
│   ├── ensemble.py        # Rank-averaging ensemble
│   └── calibration.py     # Isotonic (primary) / Platt / Temperature calibration
├── evaluation/
│   └── metrics.py         # pAUC (exact competition impl), ECE, Brier
└── train.py               # End-to-end: load → features → CV → ensemble → save
configs/base.yaml           # All hyperparameters
tests/                      # 124 tests (pytest)
notebooks/                  # EDA and analysis notebooks
```
