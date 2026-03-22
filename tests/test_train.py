"""Tests for the Trainer (training pipeline orchestrator).

Fast tests cover fold creation correctness (patient separation, positive count,
coverage). Slow tests exercise actual model training on synthetic data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from isic2024.train import Trainer


# ---------------------------------------------------------------------------
# Synthetic dataset large enough for 3-fold CV with patient grouping
# ---------------------------------------------------------------------------

def _make_train_df(n_patients: int = 10, lesions_per_patient: int = 30, seed: int = 0):
    """Synthetic DataFrame that mirrors the real ISIC 2024 structure.

    - n_patients × lesions_per_patient rows total
    - 1 malignant row per patient (first lesion of each patient)
    - Includes all required columns so the full pipeline runs without errors
    """
    rng = np.random.default_rng(seed)
    n = n_patients * lesions_per_patient

    patients = np.repeat([f"PAT_{i:03d}" for i in range(n_patients)], lesions_per_patient)
    target = np.zeros(n, dtype=np.int8)
    for p in range(n_patients):
        target[p * lesions_per_patient] = 1  # 1 malignant per patient

    lesion_id = np.full(n, None, dtype=object)
    for i in np.where(target == 1)[0]:
        lesion_id[i] = f"LESION_{i:04d}"

    age = rng.choice([35, 40, 45, 50, 55, 60], size=n).astype(float)

    tbp = {
        "tbp_lv_areaMM2": rng.uniform(0.5, 50.0, n),
        "tbp_lv_perimeterMM": rng.uniform(2.0, 30.0, n),
        "tbp_lv_eccentricity": rng.uniform(0.1, 0.9, n),
        "tbp_lv_norm_border": rng.uniform(0.0, 1.0, n),
        "tbp_lv_deltaLBnorm": rng.uniform(0.0, 5.0, n),
        "tbp_lv_color_std_mean": rng.uniform(0.0, 3.0, n),
        "tbp_lv_deltaA": rng.uniform(-2.0, 2.0, n),
        "tbp_lv_deltaB": rng.uniform(-2.0, 2.0, n),
        "tbp_lv_deltaL": rng.uniform(-5.0, 5.0, n),
        "tbp_lv_H": rng.uniform(0.0, 360.0, n),
        "tbp_lv_Hext": rng.uniform(0.0, 360.0, n),
        "tbp_lv_A": rng.uniform(-5.0, 5.0, n),
        "tbp_lv_Aext": rng.uniform(-5.0, 5.0, n),
        "tbp_lv_B": rng.uniform(-5.0, 5.0, n),
        "tbp_lv_Bext": rng.uniform(-5.0, 5.0, n),
        "tbp_lv_L": rng.uniform(20.0, 80.0, n),
        "tbp_lv_Lext": rng.uniform(20.0, 80.0, n),
        "tbp_lv_symm_2axis": rng.uniform(0.0, 1.0, n),
        "tbp_lv_nevi_confidence": rng.uniform(0.0, 1.0, n),
        "tbp_lv_dnn_lesion_confidence": rng.uniform(0.0, 1.0, n),
        "tbp_lv_norm_color": rng.uniform(0.0, 1.0, n),
        "tbp_lv_location_simple": rng.choice(["head", "trunk", "arm", "leg"], n),
    }

    return pd.DataFrame({
        "isic_id": [f"ISIC_{i:07d}" for i in range(n)],
        "patient_id": patients,
        "target": target,
        "age_approx": age,
        "sex": rng.choice(["male", "female"], n),
        "anatom_site_general": rng.choice(
            ["torso", "lower extremity", "upper extremity", "head/neck"], n
        ),
        "clin_size_long_diam_mm": rng.uniform(1.0, 20.0, n),
        "attribution": rng.choice(["Hospital_A", "Hospital_B"], n),
        "tbp_tile_type": rng.choice(["3D: white", "3D: XP"], n, p=[0.7, 0.3]),
        "image_type": "TBP tile: ISIC",
        "copyright_license": "CC-BY-NC",
        "lesion_id": lesion_id,
        "mel_thick_mm": np.where(target == 1, rng.uniform(0.5, 3.0, n), np.nan),
        "mel_mitotic_index": np.where(target == 1, "0/mm^2", None),
        "iddx_full": np.where(target == 1, "Melanoma Invasive", None),
        "iddx_1": np.where(target == 1, "Melanoma", None),
        "iddx_2": np.full(n, np.nan),
        **tbp,
    })


@pytest.fixture(scope="module")
def small_train_df():
    return _make_train_df(n_patients=10, lesions_per_patient=30)


@pytest.fixture(scope="module")
def three_fold_config(base_config):
    """Config with 3 folds and seed averaging disabled for faster tests."""
    import copy
    cfg = copy.deepcopy(base_config)
    cfg.cv.n_splits = 3
    cfg.seed_averaging.enabled = False
    cfg.lgbm.n_estimators = 50
    cfg.lgbm.early_stopping_rounds = 10
    cfg.xgb.n_estimators = 50
    cfg.xgb.early_stopping_rounds = 10
    cfg.catboost.iterations = 50
    cfg.catboost.early_stopping_rounds = 10
    return cfg


# ---------------------------------------------------------------------------
# Test 1: create_folds returns the correct number of folds
# ---------------------------------------------------------------------------

def test_create_folds_count(small_train_df, three_fold_config, tmp_path):
    trainer = Trainer(three_fold_config, output_dir=tmp_path)
    folds = trainer.create_folds(small_train_df)
    assert len(folds) == three_fold_config.cv.n_splits


# ---------------------------------------------------------------------------
# Test 2: no patient appears in both train and val of the same fold
# ---------------------------------------------------------------------------

def test_create_folds_no_patient_overlap(small_train_df, three_fold_config, tmp_path):
    trainer = Trainer(three_fold_config, output_dir=tmp_path)
    folds = trainer.create_folds(small_train_df)

    groups = small_train_df["patient_id"].values
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_patients = set(groups[train_idx])
        val_patients = set(groups[val_idx])
        overlap = train_patients & val_patients
        assert len(overlap) == 0, (
            f"Fold {fold_idx}: patients {overlap} appear in both train and val"
        )


# ---------------------------------------------------------------------------
# Test 3: each fold has at least 1 positive in val
# ---------------------------------------------------------------------------

def test_create_folds_positives_in_val(small_train_df, three_fold_config, tmp_path):
    trainer = Trainer(three_fold_config, output_dir=tmp_path)
    folds = trainer.create_folds(small_train_df)

    y = small_train_df["target"].values
    for fold_idx, (_, val_idx) in enumerate(folds):
        n_pos = int(y[val_idx].sum())
        assert n_pos >= 1, f"Fold {fold_idx} has 0 positives in val set"


# ---------------------------------------------------------------------------
# Test 4: val indices cover all rows exactly once (no gaps, no overlap)
# ---------------------------------------------------------------------------

def test_create_folds_full_coverage(small_train_df, three_fold_config, tmp_path):
    trainer = Trainer(three_fold_config, output_dir=tmp_path)
    folds = trainer.create_folds(small_train_df)

    all_val_idx = np.concatenate([val_idx for _, val_idx in folds])
    assert len(np.unique(all_val_idx)) == len(small_train_df), (
        "Val indices do not cover all rows exactly once"
    )


# ---------------------------------------------------------------------------
# Test 5: train_cv returns OOF predictions of correct shape and range
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_train_cv_oof_shape_and_range(small_train_df, three_fold_config, tmp_path):
    trainer = Trainer(three_fold_config, output_dir=tmp_path)
    folds = trainer.create_folds(small_train_df)
    results = trainer.train_cv(small_train_df, folds)

    oof_df = results["oof_df"]
    assert len(oof_df) == len(small_train_df)

    for col in ["lgbm", "xgb", "catboost", "svm", "ensemble"]:
        assert col in oof_df.columns, f"Missing OOF column: {col}"
        vals = oof_df[col].values
        assert vals.min() >= 0.0 and vals.max() <= 1.0, (
            f"{col} predictions outside [0, 1]"
        )

    # isic_id should be present since it's in the raw df
    assert "isic_id" in oof_df.columns


# ---------------------------------------------------------------------------
# Test 6: cv_results has expected structure and valid pAUC range
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_train_cv_results_structure(small_train_df, three_fold_config, tmp_path):
    trainer = Trainer(three_fold_config, output_dir=tmp_path)
    folds = trainer.create_folds(small_train_df)
    results = trainer.train_cv(small_train_df, folds)

    cv = results["cv_results"]
    for model_name in ["lgbm", "xgb", "catboost", "svm"]:
        assert model_name in cv
        assert "mean_pauc" in cv[model_name]
        assert "std_pauc" in cv[model_name]
        assert 0.0 <= cv[model_name]["mean_pauc"] <= 0.2

    assert "ensemble" in cv
    assert "pauc" in cv["ensemble"]["oof_metrics"]


# ---------------------------------------------------------------------------
# Test 7: Trainer.run smoke test — creates all output files
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_trainer_run_creates_output_files(small_train_df, three_fold_config, tmp_path):
    trainer = Trainer(three_fold_config, output_dir=tmp_path)
    results = trainer.run(small_train_df.copy())

    assert (tmp_path / "oof_predictions.csv").exists()
    assert (tmp_path / "cv_results.json").exists()
    assert (tmp_path / "feature_names.json").exists()
    assert (tmp_path / "preprocessor.pkl").exists()
    assert (tmp_path / "selector.pkl").exists()

    assert len(results["feature_names"]) > 0
