"""Tests for FeatureSelector and build_feature_pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from isic2024.features.pipeline import build_feature_pipeline
from isic2024.features.selection import FeatureSelector

# ---------------------------------------------------------------------------
# Helper: build a small controlled DataFrame
# ---------------------------------------------------------------------------

def _make_selection_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """DataFrame with known structure for testing each filter step."""
    rng = np.random.default_rng(seed)
    target = rng.integers(0, 2, size=n).astype(np.int8)

    feat_a = rng.normal(0, 1, n)  # genuinely informative
    # feat_b is highly correlated with feat_a but less predictive of target
    feat_b = feat_a * 0.99 + rng.normal(0, 0.01, n)
    # target is more correlated with feat_a than feat_b by construction above
    # (feat_a has a small extra signal added)
    feat_a = feat_a + target * 0.5

    feat_c = rng.normal(0, 1, n)   # independent feature

    const_col = np.ones(n)                           # zero variance → constant
    quasi_col = np.where(rng.random(n) < 0.996, 0.0, 1.0)  # 99.6% zeros

    return pd.DataFrame(
        {
            "patient_id": [f"P{i % 20}" for i in range(n)],
            "target": target,
            "feat_a": feat_a,
            "feat_b": feat_b,
            "feat_c": feat_c,
            "const_col": const_col,
            "quasi_col": quasi_col,
        }
    )


# ---------------------------------------------------------------------------
# Test 1: constant column removed
# ---------------------------------------------------------------------------

def test_constant_column_removed(base_config) -> None:
    df = _make_selection_df()
    selector = FeatureSelector(base_config.features)
    selector.fit(df)

    assert "const_col" not in selector.selected_cols_


# ---------------------------------------------------------------------------
# Test 2: quasi-constant column removed
# ---------------------------------------------------------------------------

def test_quasi_constant_removed(base_config) -> None:
    df = _make_selection_df()
    selector = FeatureSelector(base_config.features)
    selector.fit(df)

    assert "quasi_col" not in selector.selected_cols_


# ---------------------------------------------------------------------------
# Test 3: correlated pair — lower target-corr member is dropped
# ---------------------------------------------------------------------------

def test_correlated_pair_drops_weaker(base_config) -> None:
    df = _make_selection_df()

    # Verify feat_a and feat_b are highly correlated (|r| > 0.90)
    r = df[["feat_a", "feat_b"]].corr().loc["feat_a", "feat_b"]
    assert abs(r) > 0.90, f"Test setup error: |r|={abs(r):.3f} not > 0.90"

    selector = FeatureSelector(base_config.features)
    selector.fit(df)

    # feat_a has higher target correlation → feat_b should be dropped
    corr_a = abs(df["feat_a"].corr(df["target"]))
    corr_b = abs(df["feat_b"].corr(df["target"]))
    assert corr_a > corr_b, "Test setup: feat_a should be more correlated with target"

    assert "feat_a" in selector.selected_cols_
    assert "feat_b" not in selector.selected_cols_


# ---------------------------------------------------------------------------
# Test 4: no NaN in transform output
# ---------------------------------------------------------------------------

def test_no_nan_in_transform(base_config) -> None:
    df = _make_selection_df()
    selector = FeatureSelector(base_config.features)
    selector.fit(df)
    out = selector.transform(df)

    assert not out.isna().any().any()


# ---------------------------------------------------------------------------
# Test 5: patient_id and target not in selected_cols_
# ---------------------------------------------------------------------------

def test_patient_id_and_target_excluded(base_config) -> None:
    df = _make_selection_df()
    selector = FeatureSelector(base_config.features)
    selector.fit(df)

    assert "patient_id" not in selector.selected_cols_
    assert "target" not in selector.selected_cols_


# ---------------------------------------------------------------------------
# Test 6: transform before fit raises RuntimeError
# ---------------------------------------------------------------------------

def test_transform_before_fit_raises(base_config) -> None:
    df = _make_selection_df()
    selector = FeatureSelector(base_config.features)

    with pytest.raises(RuntimeError, match="fit"):
        selector.transform(df)


# ---------------------------------------------------------------------------
# Test 7: train/val consistency — same columns applied to val
# ---------------------------------------------------------------------------

def test_train_val_consistency(base_config) -> None:
    df_train = _make_selection_df(n=200, seed=0)
    df_val = _make_selection_df(n=50, seed=99)

    selector = FeatureSelector(base_config.features)
    selector.fit(df_train)

    # Transform val should succeed and use train-fitted column list
    out_val = selector.transform(df_val)
    assert list(out_val.columns) == selector.selected_cols_
    assert out_val.shape == (50, len(selector.selected_cols_))


# ---------------------------------------------------------------------------
# Test 8: build_feature_pipeline returns 4-tuple; feature count reasonable
# ---------------------------------------------------------------------------

def test_pipeline_roundtrip(base_config, synthetic_df) -> None:
    df_out, feature_names, preprocessor, selector = build_feature_pipeline(
        synthetic_df.copy(), base_config, is_train=True
    )

    # Returns the correct types
    assert isinstance(df_out, pd.DataFrame)
    assert isinstance(feature_names, list)
    assert len(feature_names) > 0

    # Selected features must all be present in df_out
    assert all(c in df_out.columns for c in feature_names)

    # patient_id and target preserved in full df_out
    assert "patient_id" in df_out.columns
    assert "target" in df_out.columns

    # Feature count is strictly less than total columns (selection did something)
    assert len(feature_names) < df_out.shape[1]

    # Returned preprocessor and selector are fitted
    assert preprocessor._fitted  # type: ignore[attr-defined]
    assert selector._is_fitted
