"""EDA-guided feature engineering for ISIC 2024 metadata.

Functions are pure transformations (no state) and operate on vectorized
pandas/numpy only. Each group is motivated by EDA findings.

Usage:
    from isic2024.features.engineering import build_features
    df_feat = build_features(df, config)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from isic2024.config import Config

# Guard against zero denominators
_EPS = 1e-8


# ---------------------------------------------------------------------------
# Color features  (ABCDE: Color)
# ---------------------------------------------------------------------------

def compute_color_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive color-based features from LAB colour-space columns.

    Columns used: tbp_lv_A/B/L (lesion), tbp_lv_Aext/Bext/Lext (skin),
    tbp_lv_H/Hext (hue), tbp_lv_deltaLBnorm.

    Returns:
        DataFrame with new colour columns added (original columns kept).
    """
    df = df.copy()

    # Euclidean LAB distance between lesion and surrounding skin
    # Established dermatology metric for colour contrast
    df["delta_e"] = np.sqrt(
        df["tbp_lv_deltaA"] ** 2
        + df["tbp_lv_deltaB"] ** 2
        + df["tbp_lv_deltaL"] ** 2
    )

    # Chroma = colourfulness (distance from grey axis in LAB)
    df["lesion_chroma"] = np.sqrt(df["tbp_lv_A"] ** 2 + df["tbp_lv_B"] ** 2)
    df["skin_chroma"] = np.sqrt(df["tbp_lv_Aext"] ** 2 + df["tbp_lv_Bext"] ** 2)
    df["delta_chroma"] = df["lesion_chroma"] - df["skin_chroma"]

    # Relative luminance: lesion vs skin
    df["color_ratio_L"] = df["tbp_lv_L"] / (df["tbp_lv_Lext"] + _EPS)

    # Hue difference between lesion centre and its periphery
    df["hue_diff"] = df["tbp_lv_H"] - df["tbp_lv_Hext"]

    logger.debug("Color features added: delta_e, lesion_chroma, skin_chroma, "
                 "delta_chroma, color_ratio_L, hue_diff")
    return df


# ---------------------------------------------------------------------------
# Shape features  (ABCDE: Asymmetry, Border, Diameter)
# ---------------------------------------------------------------------------

def compute_shape_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive shape-based features from area/perimeter columns.

    Columns used: tbp_lv_areaMM2, tbp_lv_perimeterMM.

    Returns:
        DataFrame with new shape columns added.
    """
    df = df.copy()

    # Circularity: 1.0 = perfect circle, < 1 = irregular border
    df["circularity"] = (
        4 * np.pi * df["tbp_lv_areaMM2"]
        / (df["tbp_lv_perimeterMM"] ** 2 + _EPS)
    )

    # Border complexity: long perimeter relative to area = ragged border
    df["border_complexity"] = df["tbp_lv_perimeterMM"] / (
        np.sqrt(df["tbp_lv_areaMM2"]) + _EPS
    )

    # Log-area: compresses the heavy right tail from large lesions
    df["log_area"] = np.log1p(df["tbp_lv_areaMM2"])

    logger.debug("Shape features added: circularity, border_complexity, log_area")
    return df


# ---------------------------------------------------------------------------
# Interaction features  (clinically meaningful cross-terms)
# ---------------------------------------------------------------------------

def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive interaction terms motivated by EDA and clinical literature.

    Only high-value interactions are included to avoid feature bloat.

    Note:
        Requires ``compute_color_features`` to have been called first —
        ``delta_e`` must already be present in df. ``build_features`` enforces
        this ordering automatically.

    Returns:
        DataFrame with new interaction columns added.
    """
    df = df.copy()

    # Older patients with larger lesions: age-size interaction is clinically
    # meaningful (melanoma risk increases with age AND lesion size)
    df["age_x_area"] = df["age_approx"] * df["tbp_lv_areaMM2"]

    # Shape × colour: irregular shape combined with high colour contrast
    # (both are ABCDE warning signs independently; their product amplifies signal)
    df["eccentricity_x_delta_e"] = df["tbp_lv_eccentricity"] * df["delta_e"]

    logger.debug("Interaction features added: age_x_area, eccentricity_x_delta_e")
    return df


# ---------------------------------------------------------------------------
# Location / count features  (ugly-duckling precursor)
# ---------------------------------------------------------------------------

def compute_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive patient-level and site-level lesion count features.

    These are simple aggregates; the full ugly-duckling z-scores are in
    ugly_duckling.py (Step 5).

    Requires columns: patient_id, anatom_site_general.

    Returns:
        DataFrame with lesion-count columns added.
    """
    df = df.copy()

    # Total lesions per patient (denominator for ugly duckling)
    df["n_lesions_patient"] = df.groupby("patient_id")["patient_id"].transform("count")

    # Lesions per (patient, anatomical site) — within-site context
    df["n_lesions_site"] = df.groupby(["patient_id", "anatom_site_general"])[
        "patient_id"
    ].transform("count")

    logger.debug("Location features added: n_lesions_patient, n_lesions_site")
    return df


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Chain all feature groups and return the enriched DataFrame.

    Applies groups controlled by config.features flags:
    - use_color       → compute_color_features
    - use_shape       → compute_shape_features
    - use_interaction → compute_interaction_features (requires color + shape output)
    - use_location    → compute_location_features

    Args:
        df: Preprocessed DataFrame (output of Preprocessor.fit_transform or .transform).
        config: Root Config object.

    Returns:
        DataFrame with all enabled engineered features appended.
    """
    n_start = len(df.columns)
    feat_cfg = config.features

    if feat_cfg.use_color:
        df = compute_color_features(df)
    if feat_cfg.use_shape:
        df = compute_shape_features(df)
    if feat_cfg.use_interaction:
        # interaction features depend on delta_e (color) and area (shape)
        if "delta_e" not in df.columns or "tbp_lv_areaMM2" not in df.columns:
            logger.warning(
                "Skipping interaction features — delta_e or tbp_lv_areaMM2 absent. "
                "Enable use_color and use_shape first."
            )
        else:
            df = compute_interaction_features(df)
    if feat_cfg.use_location:
        df = compute_location_features(df)

    n_added = len(df.columns) - n_start
    logger.info(f"build_features: added {n_added} engineered features "
                f"(total columns now {len(df.columns)})")
    return df
