"""Ugly duckling patient-wise z-score and ECDF features.

Implements the "ugly duckling" heuristic from dermatology: a lesion that looks
significantly different from a patient's other moles is more suspicious. This
module computes per-lesion z-scores relative to three grouping levels (from the
2nd place ISIC 2024 solution):

  1. [patient_id]                          — weird vs patient's other moles?
  2. [patient_id, anatom_site_general]     — weird for this body region?
  3. [patient_id, tbp_lv_location_simple]  — weird for this exact TBP location?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from isic2024.config import Config

_EPS: float = 1e-8

# Columns excluded from z-score input even if numeric dtype (post-preprocessing
# label-encoded categoricals end up as int16).
_EXCLUDE_FROM_Z: frozenset[str] = frozenset(
    {
        "tbp_lv_location_simple",
        "target",
        "has_lesion_id",
        "age_approx_missing",
    }
)

# Non-tbp_lv_* numeric columns that are meaningful for ugly duckling.
_EXTRA_NUMERIC: list[str] = ["age_approx", "clin_size_long_diam_mm"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _select_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric tbp_lv_* columns plus clinical extras suitable for z-scores.

    Excludes label-encoded categoricals (tbp_lv_location_simple) and binary
    indicator columns (has_lesion_id, age_approx_missing).
    """
    tbp_numeric = [
        c
        for c in df.columns
        if c.startswith("tbp_lv_")
        and c not in _EXCLUDE_FROM_Z
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    extra = [
        c
        for c in _EXTRA_NUMERIC
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    return tbp_numeric + extra


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_ugly_duckling(
    df: pd.DataFrame,
    group_cols: list[str],
    feature_cols: list[str],
    suffix: str,
) -> pd.DataFrame:
    """Add per-lesion z-scores and ECDF ranks relative to a patient group.

    For each feature f and each group g defined by group_cols:
        z_score  = (f - group_mean(f)) / (group_std(f) + eps)
        ecdf     = rank(f, pct=True) within group

    Edge cases:
        - Single-member group: std = 0, so z = 0.0 (denominator is eps).
        - Single-member group: ecdf = 0.5 (centred, semantically neutral).

    Args:
        df: Input DataFrame (copy is made; original is not mutated).
        group_cols: Column names defining the grouping (e.g. ["patient_id"]).
        feature_cols: Numeric columns to compute z-scores/ECDF for.
        suffix: String appended to new column names, e.g. "pat".

    Returns:
        New DataFrame with added ``{feat}_z_{suffix}`` and ``{feat}_ecdf_{suffix}`` cols.
    """
    out = df.copy()

    group_mean = df.groupby(group_cols)[feature_cols].transform("mean")
    # std(ddof=1) returns NaN for single-member groups — fill with 0 so the
    # denominator is just _EPS and z-score becomes 0.0.
    group_std = df.groupby(group_cols)[feature_cols].transform("std").fillna(0.0)

    z = (df[feature_cols].values - group_mean.values) / (group_std.values + _EPS)
    z_df = pd.DataFrame(
        z,
        index=df.index,
        columns=[f"{c}_z_{suffix}" for c in feature_cols],
        dtype=np.float64,
    )

    ecdf = df.groupby(group_cols)[feature_cols].transform(
        lambda x: x.rank(pct=True)
    )
    # Single-member groups: rank(pct=True) returns 1.0 — replace with neutral 0.5.
    group_size = df.groupby(group_cols)[feature_cols[0]].transform("count")
    single_mask = (group_size == 1).values
    if single_mask.any():
        ecdf_values = ecdf.values.copy()
        ecdf_values[single_mask] = 0.5
        ecdf = pd.DataFrame(ecdf_values, index=df.index, columns=feature_cols)

    ecdf.columns = pd.Index([f"{c}_ecdf_{suffix}" for c in feature_cols])

    out = pd.concat([out, z_df, ecdf], axis=1)
    return out


def build_ugly_duckling_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Compute ugly duckling features at all three grouping levels.

    Runs compute_ugly_duckling for:
      1. [patient_id]                           → suffix "pat"
      2. [patient_id, anatom_site_general]      → suffix "pat_site"
      3. [patient_id, tbp_lv_location_simple]   → suffix "pat_loc"

    Then adds three aggregate columns (patient-level):
      - ud_outlier_count:    number of pat-level z-scores with |z| > 2.0
      - ud_mean_abs_zscore:  mean absolute pat-level z-score across all features
      - is_single_lesion_patient: 1 if the patient has exactly 1 lesion

    Args:
        df: Preprocessed (and optionally engineered) DataFrame. NaN should be
            filled before calling this function.
        config: Project config (reserved for future threshold configurability).

    Returns:
        DataFrame with all ugly duckling columns added.
    """
    feature_cols = _select_feature_cols(df)
    logger.info(f"Ugly duckling: {len(feature_cols)} input features")

    out = compute_ugly_duckling(df, ["patient_id"], feature_cols, "pat")
    out = compute_ugly_duckling(
        out, ["patient_id", "anatom_site_general"], feature_cols, "pat_site"
    )
    out = compute_ugly_duckling(
        out, ["patient_id", "tbp_lv_location_simple"], feature_cols, "pat_loc"
    )

    # Aggregate over patient-level z-scores only.
    pat_z_cols = [f"{c}_z_pat" for c in feature_cols]
    abs_z = out[pat_z_cols].abs()
    out["ud_outlier_count"] = (abs_z > 2.0).sum(axis=1).astype(np.int16)
    out["ud_mean_abs_zscore"] = abs_z.mean(axis=1)

    # Single-lesion patient flag.
    if "n_lesions_patient" in out.columns:
        out["is_single_lesion_patient"] = (out["n_lesions_patient"] == 1).astype(np.int8)
    else:
        lesion_counts = out.groupby("patient_id")["patient_id"].transform("count")
        out["is_single_lesion_patient"] = (lesion_counts == 1).astype(np.int8)

    n_new = out.shape[1] - df.shape[1]
    logger.info(f"Ugly duckling: added {n_new} columns (total: {out.shape[1]})")
    return out
