"""Tests for ugly duckling patient-wise z-score features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from isic2024.features.ugly_duckling import (
    _select_feature_cols,
    build_ugly_duckling_features,
    compute_ugly_duckling,
)

# ---------------------------------------------------------------------------
# Helper: minimal DataFrame for isolated tests
# ---------------------------------------------------------------------------

def _make_minimal_df(n_patients: int = 3, lesions_per_patient: int = 5) -> pd.DataFrame:
    """Minimal DataFrame with known structure for z-score verification."""
    rng = np.random.default_rng(0)
    n = n_patients * lesions_per_patient
    patient_ids = [f"P{i}" for i in range(n_patients)]
    patients = np.repeat(patient_ids, lesions_per_patient)
    _site_pool = ["arm", "leg", "trunk", "head", "leg"]
    sites = np.array(_site_pool * (n // len(_site_pool) + 1))[:n]
    return pd.DataFrame(
        {
            "patient_id": patients,
            "anatom_site_general": sites,
            "tbp_lv_location_simple": sites,
            "tbp_lv_areaMM2": rng.uniform(1.0, 20.0, n),
            "tbp_lv_eccentricity": rng.uniform(0.1, 0.9, n),
            "age_approx": rng.uniform(30.0, 70.0, n),
            "clin_size_long_diam_mm": rng.uniform(1.0, 15.0, n),
        }
    )


def _make_single_lesion_df() -> pd.DataFrame:
    """DataFrame where PAT_X has only 1 lesion (for single-member group tests)."""
    multi = _make_minimal_df(n_patients=3, lesions_per_patient=5)
    single = pd.DataFrame(
        {
            "patient_id": ["PAT_X"],
            "anatom_site_general": ["trunk"],
            "tbp_lv_location_simple": ["trunk"],
            "tbp_lv_areaMM2": [10.0],
            "tbp_lv_eccentricity": [0.5],
            "age_approx": [45.0],
            "clin_size_long_diam_mm": [5.0],
        }
    )
    return pd.concat([multi, single], ignore_index=True)


# ---------------------------------------------------------------------------
# Test 1: z-scores mean ≈ 0 within each patient group
# ---------------------------------------------------------------------------

def test_z_scores_mean_zero_within_group() -> None:
    df = _make_minimal_df()
    feature_cols = ["tbp_lv_areaMM2", "tbp_lv_eccentricity"]
    out = compute_ugly_duckling(df, ["patient_id"], feature_cols, "pat")

    for patient in df["patient_id"].unique():
        mask = out["patient_id"] == patient
        for feat in feature_cols:
            group_mean = out.loc[mask, f"{feat}_z_pat"].mean()
            assert abs(group_mean) < 1e-10, (
                f"Patient {patient}, feature {feat}: mean z-score = {group_mean:.2e}, expected ≈ 0"
            )


# ---------------------------------------------------------------------------
# Test 2: single-member patient group → z-score = 0.0
# ---------------------------------------------------------------------------

def test_single_member_group_z_zero() -> None:
    df = _make_single_lesion_df()
    feature_cols = ["tbp_lv_areaMM2", "tbp_lv_eccentricity"]
    out = compute_ugly_duckling(df, ["patient_id"], feature_cols, "pat")

    single_row = out[out["patient_id"] == "PAT_X"]
    assert len(single_row) == 1
    for feat in feature_cols:
        z_val = single_row[f"{feat}_z_pat"].iloc[0]
        assert abs(z_val) < 1e-6, f"Single-lesion z-score = {z_val:.2e}, expected 0.0"


# ---------------------------------------------------------------------------
# Test 3: single-member group → ecdf = 0.5
# ---------------------------------------------------------------------------

def test_single_member_group_ecdf_neutral() -> None:
    df = _make_single_lesion_df()
    feature_cols = ["tbp_lv_areaMM2", "tbp_lv_eccentricity"]
    out = compute_ugly_duckling(df, ["patient_id"], feature_cols, "pat")

    single_row = out[out["patient_id"] == "PAT_X"]
    for feat in feature_cols:
        ecdf_val = single_row[f"{feat}_ecdf_pat"].iloc[0]
        assert ecdf_val == pytest.approx(0.5), (
            f"Single-lesion ecdf = {ecdf_val}, expected 0.5"
        )


# ---------------------------------------------------------------------------
# Test 4: ud_outlier_count flags planted extreme
# ---------------------------------------------------------------------------

def test_outlier_count_detects_extreme(base_config) -> None:
    df = _make_minimal_df(n_patients=2, lesions_per_patient=10)
    # Plant an extreme value for patient P0, row 0
    df.loc[0, "tbp_lv_areaMM2"] = 10_000.0
    out = build_ugly_duckling_features(df, base_config)

    # Row 0 should have at least 1 outlier flag (the extreme areaMM2)
    assert out.loc[0, "ud_outlier_count"] >= 1


# ---------------------------------------------------------------------------
# Test 5: is_single_lesion_patient = 1 for single-lesion patient, 0 for multi
# ---------------------------------------------------------------------------

def test_is_single_lesion_patient(base_config) -> None:
    df = _make_single_lesion_df()
    out = build_ugly_duckling_features(df, base_config)

    single_flag = out.loc[out["patient_id"] == "PAT_X", "is_single_lesion_patient"].iloc[0]
    multi_flag = out.loc[out["patient_id"] == "P0", "is_single_lesion_patient"].iloc[0]

    assert single_flag == 1
    assert multi_flag == 0


# ---------------------------------------------------------------------------
# Test 6: no NaN or inf in output
# ---------------------------------------------------------------------------

def test_no_nan_or_inf(base_config) -> None:
    df = _make_minimal_df()
    out = build_ugly_duckling_features(df, base_config)

    # Check all new numeric columns added by ugly duckling
    new_cols = [
        c for c in out.columns
        if "_z_" in c or "_ecdf_" in c
        or c.startswith("ud_") or c == "is_single_lesion_patient"
    ]
    assert new_cols, "No ugly duckling columns found"

    for col in new_cols:
        if pd.api.types.is_numeric_dtype(out[col]):
            assert not out[col].isna().any(), f"NaN found in {col}"
            assert not np.isinf(out[col].values).any(), f"Inf found in {col}"


# ---------------------------------------------------------------------------
# Test 7: input DataFrame is not mutated
# ---------------------------------------------------------------------------

def test_input_not_mutated() -> None:
    df = _make_minimal_df()
    original_cols = list(df.columns)
    original_shape = df.shape
    feature_cols = ["tbp_lv_areaMM2"]
    _ = compute_ugly_duckling(df, ["patient_id"], feature_cols, "pat")

    assert list(df.columns) == original_cols
    assert df.shape == original_shape


# ---------------------------------------------------------------------------
# Test 8: correct column count
# ---------------------------------------------------------------------------

def test_column_count(base_config) -> None:
    df = _make_minimal_df()
    feature_cols = _select_feature_cols(df)
    n_features = len(feature_cols)
    n_levels = 3
    n_types = 2  # z + ecdf
    n_aggregates = 3  # ud_outlier_count, ud_mean_abs_zscore, is_single_lesion_patient

    out = build_ugly_duckling_features(df, base_config)
    expected_new = n_features * n_levels * n_types + n_aggregates
    actual_new = out.shape[1] - df.shape[1]

    assert actual_new == expected_new, (
        f"Expected {expected_new} new columns, got {actual_new}"
    )


# ---------------------------------------------------------------------------
# Test 9: all _z_pat columns are float64
# ---------------------------------------------------------------------------

def test_z_columns_are_float64(base_config) -> None:
    df = _make_minimal_df()
    out = build_ugly_duckling_features(df, base_config)

    z_cols = [c for c in out.columns if c.endswith("_z_pat")]
    assert z_cols, "No _z_pat columns found"
    for col in z_cols:
        assert out[col].dtype == np.float64, f"{col} dtype = {out[col].dtype}, expected float64"


# ---------------------------------------------------------------------------
# Test 10: ud_mean_abs_zscore ≥ 0 everywhere
# ---------------------------------------------------------------------------

def test_mean_abs_zscore_nonnegative(base_config) -> None:
    df = _make_minimal_df()
    out = build_ugly_duckling_features(df, base_config)
    assert (out["ud_mean_abs_zscore"] >= 0.0).all()
