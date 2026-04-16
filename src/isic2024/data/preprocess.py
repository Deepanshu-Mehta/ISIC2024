"""Preprocessing pipeline for ISIC 2024 metadata features.

Handles leakage removal, indicator creation, imputation, and categorical encoding.
Designed for fit/transform pattern so train statistics are never leaked into validation.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder

from isic2024.config import DataConfig


# Columns to always drop — purely administrative, zero predictive value
_ALWAYS_DROP = ["isic_id", "image_type", "copyright_license"]

# ⚠️  NEAR-LEAKAGE: lesion_id is non-null ONLY for biopsied lesions.
# EDA confirmed ALL 393 malignant cases have lesion_id → INFINITE lift.
# We create has_lesion_id binary indicator and then drop the raw ID.
_LESION_ID_COL = "lesion_id"

# Leakage regex — covers iddx_full, iddx_1 … iddx_5
# iddx_full has BOTH value-based leakage ("Melanoma Invasive" strings)
# AND missingness-based leakage (only non-null for confirmed malignant cases).
_LEAKAGE_REGEX = re.compile(r"^iddx_")

# Categorical columns to label-encode (tbp_tile_type KEPT — mild EDA signal;
# tbp_lv_location_simple needed for ugly duckling Step-5 groupings)
_CATEGORICALS = ["sex", "anatom_site_general", "attribution", "tbp_tile_type", "tbp_lv_location_simple"]

# Imputation strategy per column
_IMPUTE_CONFIG: dict[str, str] = {
    "age_approx": "median",
    "sex": "mode",
    "anatom_site_general": "mode",
}


class Preprocessor:
    """Fit-transform preprocessor for ISIC 2024 metadata.

    Learns statistics (medians, modes, label mappings) on the training fold
    and applies identical transforms to validation/test data.

    Usage:
        prep = Preprocessor(config.data)
        X_train = prep.fit_transform(df_train)
        X_val   = prep.transform(df_val)
    """

    def __init__(self, config: DataConfig) -> None:
        self._config = config
        self._impute_values: dict[str, Any] = {}
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._missing_indicator_cols: list[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on df and return transformed copy.

        Args:
            df: Raw training DataFrame (includes target column).

        Returns:
            Preprocessed DataFrame (target column preserved if present).
        """
        df = df.copy()
        df = self._pre_indicators(df)
        df = self._drop_columns(df)
        df = self._add_missing_indicators(df, fit=True)   # before imputation
        df = self._fit_impute(df)
        df = self._fit_encode(df)
        df = self._fill_remaining_nan(df)
        self._fitted = True
        self._log_summary(df)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply previously fitted transforms to new data.

        Args:
            df: Raw validation or test DataFrame.

        Returns:
            Preprocessed DataFrame.

        Raises:
            RuntimeError: If called before fit_transform.
        """
        if not self._fitted:
            raise RuntimeError("Call fit_transform() before transform().")
        df = df.copy()
        df = self._pre_indicators(df)
        df = self._drop_columns(df)
        df = self._add_missing_indicators(df, fit=False)  # before imputation
        df = self._apply_impute(df)
        df = self._apply_encode(df)
        df = self._fill_remaining_nan(df)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pre_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived indicator columns before leakage columns are dropped.

        ⚠️ has_lesion_id: ALL malignant cases have lesion_id (INFINITE lift in EDA).
        Use only the binary indicator — raw lesion_id is dropped by _drop_columns.
        """
        if _LESION_ID_COL in df.columns:
            df["has_lesion_id"] = df[_LESION_ID_COL].notna().astype(np.int8)
            logger.debug(
                f"has_lesion_id: {int(df['has_lesion_id'].sum())} non-null "
                f"({df['has_lesion_id'].mean() * 100:.2f}% of rows)"
            )
        else:
            # Test set may not expose lesion_id — default to 0
            df["has_lesion_id"] = np.int8(0)
            logger.warning("lesion_id column absent; has_lesion_id set to 0")
        return df

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove leakage columns, non-feature metadata, and raw lesion_id."""
        # Explicit leakage: mel_thick_mm, mel_mitotic_index (from config)
        leakage_explicit = [c for c in self._config.leakage_cols if c in df.columns]
        # Regex leakage: iddx_full, iddx_1 … iddx_5
        leakage_regex = [c for c in df.columns if _LEAKAGE_REGEX.match(c)]
        # Non-informative metadata (image_type=zero variance, copyright_license)
        meta = [c for c in self._config.meta_cols if c in df.columns]
        # Administrative ID columns + raw lesion_id (replaced by has_lesion_id)
        always = [c for c in _ALWAYS_DROP if c in df.columns]
        lesion_id = [_LESION_ID_COL] if _LESION_ID_COL in df.columns else []

        to_drop = list(
            dict.fromkeys(leakage_explicit + leakage_regex + meta + always + lesion_id)
        )
        if to_drop:
            logger.debug(f"Dropping {len(to_drop)} columns: {to_drop}")
        return df.drop(columns=to_drop, errors="ignore")

    def _fit_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Learn imputation fill values from training data and apply them."""
        for col, strategy in _IMPUTE_CONFIG.items():
            if col not in df.columns:
                continue
            n_missing = int(df[col].isna().sum())
            if strategy == "median":
                fill = df[col].median()
            elif strategy == "mode":
                modes = df[col].mode()
                fill = modes.iloc[0] if len(modes) > 0 else np.nan
            else:
                raise ValueError(f"Unknown impute strategy '{strategy}' for '{col}'")
            self._impute_values[col] = fill
            if n_missing:
                logger.debug(
                    f"Imputing '{col}' ({n_missing} missing, {strategy}={fill!r})"
                )
        return self._apply_impute(df)

    def _apply_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned imputation fill values."""
        for col, fill in self._impute_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(fill)
        return df

    def _fit_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit LabelEncoders on categorical columns and encode in-place."""
        for col in _CATEGORICALS:
            if col not in df.columns:
                continue
            df[col] = df[col].fillna("__missing__").astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col]).astype(np.int16)
            self._label_encoders[col] = le
            logger.debug(f"LabelEncoded '{col}' → {len(le.classes_)} classes")
        return df

    def _apply_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted LabelEncoders; unseen categories encode to -1."""
        for col, le in self._label_encoders.items():
            if col not in df.columns:
                continue
            known = set(le.classes_)
            series = df[col].fillna("__missing__").astype(str)
            df[col] = series.map(
                lambda v, le=le, known=known: int(le.transform([v])[0]) if v in known else -1
            ).astype(np.int16)
        return df

    def _add_missing_indicators(self, df: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        """Add binary missingness indicator columns for numeric cols >1% missing.

        Indicators are created before filling remaining NaNs with 0.
        """
        numeric_cols: list[str] = [
            str(c) for c in df.select_dtypes(include=[np.number]).columns
        ]
        skip = {self._config.target_col, self._config.patient_col, "has_lesion_id"}
        numeric_cols = [c for c in numeric_cols if c not in skip]

        if fit:
            self._missing_indicator_cols = [
                c for c in numeric_cols if df[c].isna().mean() > 0.01
            ]
            if self._missing_indicator_cols:
                logger.debug(
                    f"Adding missing indicators for {len(self._missing_indicator_cols)} "
                    f"column(s): {self._missing_indicator_cols}"
                )

        for col in self._missing_indicator_cols:
            if col in df.columns:
                df[f"{col}_missing"] = df[col].isna().astype(np.int8)
                # NaN fill handled by _fit_impute (imputed cols) or _fill_remaining_nan

        return df

    def _fill_remaining_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill any NaN still present in numeric columns with 0 after imputation.

        Catches tbp_lv_* and other numeric cols not covered by _IMPUTE_CONFIG.
        Called last so it never overwrites a valid imputed value.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        remaining = [c for c in numeric_cols if df[c].isna().any()]
        if remaining:
            logger.debug(f"Filling remaining NaN with 0 for {len(remaining)} col(s): {remaining}")
            df[remaining] = df[remaining].fillna(0.0)
        return df

    def _log_summary(self, df: pd.DataFrame) -> None:
        """Log post-preprocessing shape and any remaining NaN counts."""
        feature_df = df.drop(columns=[self._config.target_col], errors="ignore")
        n_nan = int(feature_df.isna().sum().sum())
        logger.info(f"After preprocessing: shape={df.shape}, remaining NaN={n_nan}")
        if n_nan > 0:
            nan_cols = feature_df.columns[feature_df.isna().any()].tolist()
            logger.warning(f"NaN remaining in: {nan_cols}")
