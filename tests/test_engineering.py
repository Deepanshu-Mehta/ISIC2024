"""Tests for src/isic2024/features/engineering.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from isic2024.features.engineering import (
    build_features,
    compute_color_features,
    compute_interaction_features,
    compute_location_features,
    compute_shape_features,
)

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def preprocessed_df(synthetic_df: pd.DataFrame, base_config) -> pd.DataFrame:
    """Return a preprocessed version of the synthetic fixture."""
    from isic2024.data.preprocess import Preprocessor

    prep = Preprocessor(base_config.data)
    return prep.fit_transform(synthetic_df)


# ---------------------------------------------------------------------------
# Color features
# ---------------------------------------------------------------------------

class TestColorFeatures:
    def test_columns_created(self, synthetic_df, base_config):
        df = preprocessed_df(synthetic_df, base_config)
        out = compute_color_features(df)
        for col in ["delta_e", "lesion_chroma", "skin_chroma", "delta_chroma",
                    "color_ratio_L", "hue_diff"]:
            assert col in out.columns, f"Missing colour column: {col}"

    def test_delta_e_hand_calculated(self, synthetic_df, base_config):
        """delta_e = sqrt(dA² + dB² + dL²) — verify against manual calc."""
        df = preprocessed_df(synthetic_df, base_config)
        out = compute_color_features(df)

        expected = np.sqrt(
            df["tbp_lv_deltaA"] ** 2
            + df["tbp_lv_deltaB"] ** 2
            + df["tbp_lv_deltaL"] ** 2
        )
        np.testing.assert_allclose(out["delta_e"].values, expected.values, rtol=1e-6)

    def test_delta_e_non_negative(self, synthetic_df, base_config):
        df = preprocessed_df(synthetic_df, base_config)
        out = compute_color_features(df)
        assert (out["delta_e"] >= 0).all()

    def test_lesion_chroma_non_negative(self, synthetic_df, base_config):
        df = preprocessed_df(synthetic_df, base_config)
        out = compute_color_features(df)
        assert (out["lesion_chroma"] >= 0).all()

    def test_no_nan_in_color_features(self, synthetic_df, base_config):
        df = preprocessed_df(synthetic_df, base_config)
        out = compute_color_features(df)
        color_cols = ["delta_e", "lesion_chroma", "skin_chroma",
                      "delta_chroma", "color_ratio_L", "hue_diff"]
        assert out[color_cols].isna().sum().sum() == 0

    def test_no_inf_in_color_features(self, synthetic_df, base_config):
        df = preprocessed_df(synthetic_df, base_config)
        out = compute_color_features(df)
        color_cols = ["delta_e", "lesion_chroma", "skin_chroma",
                      "delta_chroma", "color_ratio_L", "hue_diff"]
        assert np.isfinite(out[color_cols].values).all()

    def test_input_not_mutated(self, synthetic_df, base_config):
        df = preprocessed_df(synthetic_df, base_config)
        original_cols = list(df.columns)
        compute_color_features(df)
        assert list(df.columns) == original_cols


# ---------------------------------------------------------------------------
# Shape features
# ---------------------------------------------------------------------------

class TestShapeFeatures:
    def test_columns_created(self, synthetic_df, base_config):
        df = preprocessed_df(synthetic_df, base_config)
        out = compute_shape_features(df)
        for col in ["circularity", "border_complexity", "log_area"]:
            assert col in out.columns, f"Missing shape column: {col}"

    def test_circularity_perfect_circle(self):
        """For a perfect circle: circularity = 4π·A / P² = 4π·(πr²) / (2πr)² = 1.0."""
        r = 5.0
        area = np.pi * r ** 2
        perimeter = 2 * np.pi * r
        df = pd.DataFrame({
            "tbp_lv_areaMM2": [area],
            "tbp_lv_perimeterMM": [perimeter],
        })
        out = compute_shape_features(df)
        assert abs(float(out["circularity"].iloc[0]) - 1.0) < 1e-4

    def test_circularity_positive(self, synthetic_df, base_config):
        """Circularity must always be positive (area and perimeter are both > 0)."""
        df = preprocessed_df(synthetic_df, base_config)
        out = compute_shape_features(df)
        assert (out["circularity"] > 0).all()

    def test_log_area_non_negative(self, synthetic_df, base_config):
        """log1p(positive area) must be >= 0."""
        df = preprocessed_df(synthetic_df, base_config)
        out = compute_shape_features(df)
        assert (out["log_area"] >= 0).all()

    def test_no_nan_in_shape_features(self, synthetic_df, base_config):
        df = preprocessed_df(synthetic_df, base_config)
        out = compute_shape_features(df)
        shape_cols = ["circularity", "border_complexity", "log_area"]
        assert out[shape_cols].isna().sum().sum() == 0

    def test_no_inf_in_shape_features(self, synthetic_df, base_config):
        df = preprocessed_df(synthetic_df, base_config)
        out = compute_shape_features(df)
        shape_cols = ["circularity", "border_complexity", "log_area"]
        assert np.isfinite(out[shape_cols].values).all()


# ---------------------------------------------------------------------------
# Interaction features
# ---------------------------------------------------------------------------

class TestInteractionFeatures:
    def _enriched_df(self, synthetic_df, base_config):
        """Return df with color + shape features already applied."""
        df = preprocessed_df(synthetic_df, base_config)
        df = compute_color_features(df)
        df = compute_shape_features(df)
        return df

    def test_columns_created(self, synthetic_df, base_config):
        df = self._enriched_df(synthetic_df, base_config)
        out = compute_interaction_features(df)
        assert "age_x_area" in out.columns
        assert "eccentricity_x_delta_e" in out.columns

    def test_age_x_area_hand_calculated(self, synthetic_df, base_config):
        df = self._enriched_df(synthetic_df, base_config)
        out = compute_interaction_features(df)
        expected = df["age_approx"] * df["tbp_lv_areaMM2"]
        np.testing.assert_allclose(out["age_x_area"].values, expected.values, rtol=1e-6)

    def test_no_nan_in_interaction_features(self, synthetic_df, base_config):
        df = self._enriched_df(synthetic_df, base_config)
        out = compute_interaction_features(df)
        assert out[["age_x_area", "eccentricity_x_delta_e"]].isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# Location features
# ---------------------------------------------------------------------------

class TestLocationFeatures:
    def test_columns_created(self, synthetic_df, base_config):
        df = preprocessed_df(synthetic_df, base_config)
        out = compute_location_features(df)
        assert "n_lesions_patient" in out.columns
        assert "n_lesions_site" in out.columns

    def test_n_lesions_patient_correct(self, synthetic_df, base_config):
        """Synthetic fixture: 5 patients × 20 lesions each → n_lesions_patient = 20 for all."""
        df = preprocessed_df(synthetic_df, base_config)
        out = compute_location_features(df)
        assert (out["n_lesions_patient"] == 20).all()

    def test_n_lesions_site_positive(self, synthetic_df, base_config):
        df = preprocessed_df(synthetic_df, base_config)
        out = compute_location_features(df)
        assert (out["n_lesions_site"] > 0).all()

    def test_n_lesions_site_lte_patient(self, synthetic_df, base_config):
        """Site count must be <= patient count (can't have more site lesions than total)."""
        df = preprocessed_df(synthetic_df, base_config)
        out = compute_location_features(df)
        assert (out["n_lesions_site"] <= out["n_lesions_patient"]).all()

    def test_no_nan_in_location_features(self, synthetic_df, base_config):
        df = preprocessed_df(synthetic_df, base_config)
        out = compute_location_features(df)
        assert out[["n_lesions_patient", "n_lesions_site"]].isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# build_features — integration
# ---------------------------------------------------------------------------

class TestBuildFeatures:
    def test_all_groups_enabled(self, synthetic_df, base_config):
        """With all feature groups on, all expected columns must be present."""
        df = preprocessed_df(synthetic_df, base_config)
        out = build_features(df, base_config)
        expected = [
            "delta_e", "lesion_chroma", "skin_chroma", "delta_chroma",
            "color_ratio_L", "hue_diff",
            "circularity", "border_complexity", "log_area",
            "age_x_area", "eccentricity_x_delta_e",
            "n_lesions_patient", "n_lesions_site",
        ]
        for col in expected:
            assert col in out.columns, f"Missing engineered column: {col}"

    def test_more_columns_than_input(self, synthetic_df, base_config):
        df = preprocessed_df(synthetic_df, base_config)
        n_before = len(df.columns)
        out = build_features(df, base_config)
        assert len(out.columns) > n_before

    def test_target_preserved(self, synthetic_df, base_config):
        df = preprocessed_df(synthetic_df, base_config)
        out = build_features(df, base_config)
        assert "target" in out.columns
        assert out["target"].sum() == 3

    def test_no_nan_after_build(self, synthetic_df, base_config):
        df = preprocessed_df(synthetic_df, base_config)
        out = build_features(df, base_config)
        feature_cols = [c for c in out.columns if c != base_config.data.target_col]
        assert out[feature_cols].isna().sum().sum() == 0

    def test_no_inf_after_build(self, synthetic_df, base_config):
        df = preprocessed_df(synthetic_df, base_config)
        out = build_features(df, base_config)
        numeric = out.select_dtypes(include=[np.number])
        assert np.isfinite(numeric.values).all()
