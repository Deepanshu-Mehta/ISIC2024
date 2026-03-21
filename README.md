# ISIC 2024 — Phase 1: Skin Cancer Detection (Tabular ML)

Binary classification of malignant vs benign skin lesions using **metadata features only** from the [ISIC 2024 SLICE-3D dataset](https://www.kaggle.com/competitions/isic-2024-challenge). No images in Phase 1.

## Why metadata-only works
The `tbp_lv_*` columns are machine-extracted measurements from Canfield's Vectra Total Body Photography system, numerically encoding the ABCDE dermatology criteria (Asymmetry, Border, Color, Diameter, Evolution). We are effectively using quantified image features.

## Approach
- **Models**: LightGBM + XGBoost + CatBoost ensemble + SVM baseline
- **Ensemble**: Rank-averaging across all GBDT models
- **Feature engineering**: Color, shape, interaction features + patient-wise "ugly duckling" z-scores at 3 grouping levels
- **CV**: `StratifiedGroupKFold` — patient_id never leaks across folds
- **Metric**: Partial AUC above 80% TPR (pAUC ∈ [0.0, 0.2])
- **Stability**: Multi-seed averaging (seeds: 42, 123, 456, 789, 2024)

## Dataset
- 401,059 lesion records, 393 confirmed malignant (0.098% positive rate)
- Extreme class imbalance handled via `scale_pos_weight` in GBDT models
- Data not included in this repo — download from [Kaggle](https://www.kaggle.com/competitions/isic-2024-challenge/data)

## Setup
```bash
conda env create -f environment.yml
conda activate isic2024
pip install -e .
```

## Usage
```bash
# Run tests
python -m pytest tests/ -v --tb=short

# Lint / format
ruff check src/ tests/
ruff format src/ tests/

# Launch notebooks
jupyter lab

# Train pipeline
python -m src.isic2024.train --config configs/base.yaml
```

## Project Structure
```
src/isic2024/
├── config.py              # Config dataclasses + YAML loading
├── data/
│   ├── loader.py          # CSV loading + column validation
│   └── preprocess.py      # Imputation, encoding, leakage removal
├── features/
│   ├── engineering.py     # Color, shape, interaction features
│   ├── ugly_duckling.py   # Patient-wise z-scores
│   ├── selection.py       # Correlation + variance filtering
│   └── pipeline.py        # Full feature pipeline
├── models/
│   ├── gbdt.py            # LightGBM / XGBoost / CatBoost wrappers
│   ├── svm_baseline.py    # SVM + Platt scaling
│   ├── ensemble.py        # Rank-averaging ensemble
│   └── calibration.py     # Isotonic / Platt / Temperature calibration
├── evaluation/
│   ├── metrics.py         # pAUC (exact competition implementation), ECE
│   └── plots.py           # Reliability diagrams, score distributions
├── optimize.py            # Optuna hyperparameter search
└── train.py               # End-to-end training orchestrator
notebooks/
configs/base.yaml
tests/
```

## Expected Performance
| Model | pAUC |
|-------|------|
| SVM baseline | 0.10 – 0.13 |
| GBDT (no UD features) | 0.14 – 0.16 |
| GBDT + ugly duckling + seed avg | **0.16 – 0.17** |
| Phase 1 target | **≥ 0.16** |
