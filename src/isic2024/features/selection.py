"""Feature selection: constant filter, quasi-constant filter, correlation filter.

The FeatureSelector learns which columns to keep on the training fold and
reapplies the same column list at validation/test time — no re-fitting.

Correlation threshold is configurable (default 0.90). When two features are
highly correlated, the one with lower absolute correlation with the target is
dropped. This preserves the most predictive member of each correlated cluster.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from isic2024.config import FeaturesConfig

# Columns excluded from feature selection regardless of dtype.
_NON_FEATURE_COLS: frozenset[str] = frozenset({"patient_id", "target"})


class FeatureSelector:
    """Filter features by variance, quasi-constancy, and pairwise correlation.

    Usage::

        selector = FeatureSelector(config.features)
        selector.fit(df_train, target_col="target")
        df_val_features = selector.transform(df_val)
    """

    def __init__(self, config: FeaturesConfig) -> None:
        self._config = config
        self.selected_cols_: list[str] = []
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, target_col: str = "target") -> FeatureSelector:
        """Learn which features to keep from the training DataFrame.

        Args:
            df: Training DataFrame (must contain ``target_col``).
            target_col: Name of the label column (excluded from features).

        Returns:
            self (for chaining).
        """
        exclude = _NON_FEATURE_COLS | {target_col}
        feat_cols: list[str] = [
            c
            for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]
        n_start = len(feat_cols)

        # Step 1 — constant columns
        surviving = self._drop_constant(df, feat_cols)
        logger.debug(f"After constant filter: {len(surviving)} / {n_start}")

        # Step 2 — quasi-constant columns
        surviving = self._drop_quasi_constant(df, surviving)
        logger.debug(f"After quasi-constant filter: {len(surviving)} / {n_start}")

        # Step 3 — correlated pairs
        surviving = self._drop_correlated(df, surviving, target_col)
        logger.debug(f"After correlation filter: {len(surviving)} / {n_start}")

        self.selected_cols_ = surviving
        self._is_fitted = True
        logger.info(
            f"FeatureSelector: {n_start} → {len(surviving)} features "
            f"(removed {n_start - len(surviving)})"
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df restricted to the fitted feature set.

        Args:
            df: DataFrame to filter (must contain all columns in selected_cols_).

        Returns:
            DataFrame with only the selected feature columns.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "FeatureSelector.transform() called before fit(). Call fit() first."
            )
        return df[self.selected_cols_]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _drop_constant(self, df: pd.DataFrame, cols: list[str]) -> list[str]:
        variances = df[cols].var()
        return [c for c in cols if variances[c] >= self._config.variance_threshold]

    def _drop_quasi_constant(self, df: pd.DataFrame, cols: list[str]) -> list[str]:
        surviving = []
        for col in cols:
            counts = df[col].value_counts(normalize=True)
            if counts.iloc[0] <= self._config.quasi_const_threshold:
                surviving.append(col)
        return surviving

    def _drop_correlated(
        self, df: pd.DataFrame, cols: list[str], target_col: str
    ) -> list[str]:
        if len(cols) < 2:
            return cols

        threshold = self._config.correlation_threshold
        corr_matrix = df[cols].corr().abs()
        target_corr = df[cols].corrwith(df[target_col]).abs().fillna(0.0)

        to_drop: set[str] = set()
        for i, col_i in enumerate(cols):
            if col_i in to_drop:
                continue
            for j in range(i + 1, len(cols)):
                col_j = cols[j]
                if col_j in to_drop:
                    continue
                if corr_matrix.loc[col_i, col_j] > threshold:
                    # Drop the feature with lower target correlation.
                    if target_corr[col_i] >= target_corr[col_j]:
                        to_drop.add(col_j)
                    else:
                        to_drop.add(col_i)
                        break  # col_i is dropped; skip remaining j comparisons

        n_dropped = len(to_drop)
        if n_dropped:
            logger.debug(f"Correlation filter dropped {n_dropped} columns")
        return [c for c in cols if c not in to_drop]
