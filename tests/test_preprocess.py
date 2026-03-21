"""Tests for src/isic2024/data/preprocess.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from isic2024.data.preprocess import _LEAKAGE_REGEX, Preprocessor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_preprocessor(base_config) -> Preprocessor:
    return Preprocessor(base_config.data)


# ---------------------------------------------------------------------------
# Leakage column removal
# ---------------------------------------------------------------------------

def test_leakage_cols_removed(synthetic_df: pd.DataFrame, base_config) -> None:
    """mel_thick_mm, mel_mitotic_index must be absent after preprocessing."""
    prep = make_preprocessor(base_config)
    out = prep.fit_transform(synthetic_df)

    for col in ["mel_thick_mm", "mel_mitotic_index"]:
        assert col not in out.columns, f"Leakage column '{col}' still present"


def test_iddx_cols_removed(synthetic_df: pd.DataFrame, base_config) -> None:
    """All iddx_* columns must be dropped (iddx_full, iddx_1, iddx_2, etc.)."""
    prep = make_preprocessor(base_config)
    out = prep.fit_transform(synthetic_df)

    remaining_iddx = [c for c in out.columns if _LEAKAGE_REGEX.match(c)]
    assert remaining_iddx == [], f"iddx_* columns still present: {remaining_iddx}"


def test_image_type_removed(synthetic_df: pd.DataFrame, base_config) -> None:
    """image_type (zero variance) must be dropped."""
    prep = make_preprocessor(base_config)
    out = prep.fit_transform(synthetic_df)
    assert "image_type" not in out.columns


def test_isic_id_removed(synthetic_df: pd.DataFrame, base_config) -> None:
    """isic_id (identifier, not a feature) must be dropped."""
    prep = make_preprocessor(base_config)
    out = prep.fit_transform(synthetic_df)
    assert "isic_id" not in out.columns


# ---------------------------------------------------------------------------
# Kept columns (EDA-verified signal)
# ---------------------------------------------------------------------------

def test_attribution_kept(synthetic_df: pd.DataFrame, base_config) -> None:
    """attribution (hospital source) must be kept — 7.8× malignancy variation."""
    prep = make_preprocessor(base_config)
    out = prep.fit_transform(synthetic_df)
    assert "attribution" in out.columns


def test_tbp_tile_type_kept(synthetic_df: pd.DataFrame, base_config) -> None:
    """tbp_tile_type must be kept — mild EDA signal (3D:white vs 3D:XP)."""
    prep = make_preprocessor(base_config)
    out = prep.fit_transform(synthetic_df)
    assert "tbp_tile_type" in out.columns


def test_target_preserved(synthetic_df: pd.DataFrame, base_config) -> None:
    """target column must survive preprocessing."""
    prep = make_preprocessor(base_config)
    out = prep.fit_transform(synthetic_df)
    assert "target" in out.columns
    assert out["target"].sum() == 3


# ---------------------------------------------------------------------------
# has_lesion_id indicator
# ---------------------------------------------------------------------------

def test_has_lesion_id_created(synthetic_df: pd.DataFrame, base_config) -> None:
    """has_lesion_id binary indicator must be present after preprocessing."""
    prep = make_preprocessor(base_config)
    out = prep.fit_transform(synthetic_df)
    assert "has_lesion_id" in out.columns


def test_has_lesion_id_values(synthetic_df: pd.DataFrame, base_config) -> None:
    """has_lesion_id must be 1 where lesion_id was non-null, 0 elsewhere."""
    expected_non_null = int(synthetic_df["lesion_id"].notna().sum())
    prep = make_preprocessor(base_config)
    out = prep.fit_transform(synthetic_df)
    assert int(out["has_lesion_id"].sum()) == expected_non_null


def test_raw_lesion_id_dropped(synthetic_df: pd.DataFrame, base_config) -> None:
    """Raw lesion_id must be dropped after has_lesion_id indicator is created."""
    prep = make_preprocessor(base_config)
    out = prep.fit_transform(synthetic_df)
    assert "lesion_id" not in out.columns


def test_malignant_all_have_has_lesion_id(
    synthetic_df: pd.DataFrame, base_config
) -> None:
    """All malignant rows in synthetic data should have has_lesion_id=1."""
    prep = make_preprocessor(base_config)
    out = prep.fit_transform(synthetic_df)
    malignant_mask = out["target"] == 1
    assert out.loc[malignant_mask, "has_lesion_id"].eq(1).all()


# ---------------------------------------------------------------------------
# Imputation — no NaN after fit_transform
# ---------------------------------------------------------------------------

def test_no_nan_after_fit_transform(synthetic_df: pd.DataFrame, base_config) -> None:
    """No NaN values should remain in any column after fit_transform."""
    prep = make_preprocessor(base_config)
    out = prep.fit_transform(synthetic_df)
    feature_cols = [c for c in out.columns if c != base_config.data.target_col]
    n_nan = int(out[feature_cols].isna().sum().sum())
    assert n_nan == 0, f"Found {n_nan} NaN values after preprocessing"


def test_age_approx_imputed(synthetic_df: pd.DataFrame, base_config) -> None:
    """age_approx should have no NaN after imputation."""
    assert synthetic_df["age_approx"].isna().sum() > 0, "Fixture should have NaN age"
    prep = make_preprocessor(base_config)
    out = prep.fit_transform(synthetic_df)
    assert out["age_approx"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Missing indicators
# ---------------------------------------------------------------------------

def test_missing_indicator_created_for_high_missing(
    synthetic_df: pd.DataFrame, base_config
) -> None:
    """age_approx has 6% missing → age_approx_missing indicator should be created."""
    prep = make_preprocessor(base_config)
    out = prep.fit_transform(synthetic_df)
    assert "age_approx_missing" in out.columns


def test_missing_indicator_values(synthetic_df: pd.DataFrame, base_config) -> None:
    """Missing indicator should equal 1 where original value was NaN."""
    n_originally_missing = int(synthetic_df["age_approx"].isna().sum())
    prep = make_preprocessor(base_config)
    out = prep.fit_transform(synthetic_df)
    assert int(out["age_approx_missing"].sum()) == n_originally_missing


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

def test_patient_id_preserved(synthetic_df: pd.DataFrame, base_config) -> None:
    """patient_id must survive preprocessing — required for StratifiedGroupKFold and ugly duckling."""
    prep = make_preprocessor(base_config)
    out = prep.fit_transform(synthetic_df)
    assert "patient_id" in out.columns
    assert (out["patient_id"] == synthetic_df["patient_id"]).all()


def test_categoricals_are_integer_encoded(
    synthetic_df: pd.DataFrame, base_config
) -> None:
    """sex, anatom_site_general, attribution, tbp_tile_type, tbp_lv_location_simple must be integer-encoded."""
    prep = make_preprocessor(base_config)
    out = prep.fit_transform(synthetic_df)
    for col in ["sex", "anatom_site_general", "attribution", "tbp_tile_type", "tbp_lv_location_simple"]:
        assert pd.api.types.is_integer_dtype(out[col]), (
            f"'{col}' should be integer after encoding, got {out[col].dtype}"
        )


def test_unseen_category_encodes_to_minus_one(
    synthetic_df: pd.DataFrame, base_config
) -> None:
    """At transform time, unseen category values must encode to -1."""
    prep = make_preprocessor(base_config)
    prep.fit_transform(synthetic_df)

    # Build a 1-row DataFrame with an unseen attribution value
    row = synthetic_df.iloc[[0]].copy()
    row["attribution"] = "Hospital_UNSEEN"

    out = prep.transform(row)
    assert int(out["attribution"].iloc[0]) == -1


# ---------------------------------------------------------------------------
# fit/transform consistency (no data leakage)
# ---------------------------------------------------------------------------

def test_transform_reuses_fit_statistics(
    synthetic_df: pd.DataFrame, base_config
) -> None:
    """transform() must use the same imputation values as fit_transform()."""
    train = synthetic_df.iloc[:80].copy()
    val = synthetic_df.iloc[80:].copy()

    prep = make_preprocessor(base_config)
    prep.fit_transform(train)

    # Deliberately set all age to NaN in val — should be filled with train median
    train_age_median = train["age_approx"].median()
    val["age_approx"] = np.nan
    val_out = prep.transform(val)

    assert val_out["age_approx"].isna().sum() == 0
    assert (val_out["age_approx"] == train_age_median).all()


def test_transform_fails_before_fit(synthetic_df: pd.DataFrame, base_config) -> None:
    """transform() must raise RuntimeError if called before fit_transform()."""
    prep = make_preprocessor(base_config)
    with pytest.raises(RuntimeError, match="fit_transform"):
        prep.transform(synthetic_df)


def test_fit_transform_does_not_mutate_input(
    synthetic_df: pd.DataFrame, base_config
) -> None:
    """fit_transform must not modify the original DataFrame."""
    original_cols = list(synthetic_df.columns)
    original_shape = synthetic_df.shape

    prep = make_preprocessor(base_config)
    prep.fit_transform(synthetic_df)

    assert list(synthetic_df.columns) == original_cols
    assert synthetic_df.shape == original_shape
