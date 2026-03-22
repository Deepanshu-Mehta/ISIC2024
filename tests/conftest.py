"""Shared pytest fixtures for ISIC 2024 tests.

The synthetic DataFrame mirrors the real dataset's column structure so tests
can run without downloading the actual Kaggle data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from isic2024.config import Config

# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def base_config() -> Config:
    """Default Config loaded from configs/base.yaml."""
    import pathlib
    yaml_path = pathlib.Path(__file__).parents[1] / "configs" / "base.yaml"
    return Config.from_yaml(yaml_path)


# ---------------------------------------------------------------------------
# Synthetic DataFrame fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """100-row synthetic DataFrame that mimics the ISIC 2024 train-metadata structure.

    - 5 patient_ids (20 lesions each)
    - 3 malignant rows (target=1) — one per 3 different patients
    - Includes all column types: tbp_lv_*, demographics, leakage cols, lesion_id
    - lesion_id is non-null for ALL 3 malignant rows (mirrors real dataset's INFINITE lift)
    - 6 rows have null age_approx (>1% missing → triggers missing indicator)
    """
    rng = np.random.default_rng(42)
    n = 100

    patient_ids = [f"PAT_{i:04d}" for i in range(5)]
    patients = np.repeat(patient_ids, 20)

    target = np.zeros(n, dtype=np.int8)
    malignant_idx = [0, 20, 40]   # one per first 3 patients
    target[malignant_idx] = 1

    # lesion_id: non-null for all malignant rows + 5 extra benign (biopsied but benign)
    lesion_id = np.full(n, None, dtype=object)
    for i in malignant_idx:
        lesion_id[i] = f"LESION_{i:04d}"
    for i in [5, 25, 45, 65, 85]:
        lesion_id[i] = f"LESION_{i:04d}"

    # age_approx: discretised to multiples of 5 (matches SLICE-3D privacy rounding)
    age = rng.choice([25, 30, 35, 40, 45, 50, 55, 60, 65, 70], size=n).astype(float)
    age[[1, 10, 22, 50, 70, 90]] = np.nan   # 6% missing → triggers missing indicator

    tbp_cols = {
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
        # Pre-computed AI scores (inversely predictive — higher = more benign)
        "tbp_lv_nevi_confidence": rng.uniform(0.0, 1.0, n),
        "tbp_lv_dnn_lesion_confidence": rng.uniform(0.0, 1.0, n),
        "tbp_lv_norm_color": rng.uniform(0.0, 1.0, n),
        "tbp_lv_location_simple": rng.choice(["head", "trunk", "arm", "leg"], size=n),
    }

    df = pd.DataFrame(
        {
            "isic_id": [f"ISIC_{i:07d}" for i in range(n)],
            "patient_id": patients,
            "target": target,
            # Demographics
            "age_approx": age,
            "sex": rng.choice(["male", "female", None], size=n, p=[0.48, 0.48, 0.04]),
            "anatom_site_general": rng.choice(
                ["torso", "lower extremity", "upper extremity", "head/neck", None],
                size=n,
                p=[0.35, 0.25, 0.20, 0.15, 0.05],
            ),
            "clin_size_long_diam_mm": rng.uniform(1.0, 20.0, n),
            # Hospital source (EDA: 7.8× variation in malignancy rate)
            "attribution": rng.choice(
                ["Hospital_A", "Hospital_B", "Hospital_C"], size=n
            ),
            # Scan type (EDA: mild signal — keep as feature)
            "tbp_tile_type": rng.choice(["3D: white", "3D: XP"], size=n, p=[0.7, 0.3]),
            # Zero-variance metadata (EDA: all same value → drop)
            "image_type": "TBP tile: ISIC",
            "copyright_license": "CC-BY-NC",
            # Near-leakage column (raw ID → drop after creating has_lesion_id)
            "lesion_id": lesion_id,
            # Leakage columns — only non-null for confirmed malignant
            "mel_thick_mm": np.where(target == 1, rng.uniform(0.5, 3.0, n), np.nan),
            "mel_mitotic_index": np.where(
                target == 1, rng.choice(["0/mm^2", "<1/mm^2", "1/mm^2"], n), None
            ),
            "iddx_full": np.where(target == 1, "Melanoma Invasive", None),
            "iddx_1": np.where(target == 1, "Melanoma", None),
            "iddx_2": np.full(n, np.nan),
            **tbp_cols,
        }
    )
    return df
