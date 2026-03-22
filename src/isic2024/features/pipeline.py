"""Full feature pipeline: preprocess → engineer → ugly duckling → select.

Typical usage (training)::

    df_out, feature_names, preprocessor, selector = build_feature_pipeline(
        df_train, config, is_train=True
    )
    X_train = df_out[feature_names].values
    y_train = df_out["target"].values

Typical usage (test/validation)::

    df_out, feature_names, _, _ = build_feature_pipeline(
        df_test, config,
        preprocessor=fitted_preprocessor,
        selector=fitted_selector,
        is_train=False,
    )
    X_test = df_out[feature_names].values
"""
from __future__ import annotations

import pandas as pd
from loguru import logger

from isic2024.config import Config
from isic2024.data.preprocess import Preprocessor
from isic2024.features.engineering import build_features
from isic2024.features.selection import FeatureSelector
from isic2024.features.ugly_duckling import build_ugly_duckling_features


def build_feature_pipeline(
    df: pd.DataFrame,
    config: Config,
    preprocessor: Preprocessor | None = None,
    selector: FeatureSelector | None = None,
    is_train: bool = True,
) -> tuple[pd.DataFrame, list[str], Preprocessor, FeatureSelector]:
    """Run the full feature pipeline.

    Steps:
        1. Preprocess (fit_transform on train, transform on val/test)
        2. Engineering features (stateless: color, shape, interaction, location)
        3. Ugly duckling z-scores (stateless, if enabled in config)
        4. Feature selection (fit on train, transform on val/test)

    The returned DataFrame preserves all columns (including ``target`` and
    ``patient_id``). Use ``df_out[feature_names]`` to get the model input matrix.

    Args:
        df: Raw input DataFrame (pre-preprocessing).
        config: Project config.
        preprocessor: Fitted Preprocessor for val/test path. If ``None`` and
            ``is_train=True``, a new one is created and fitted.
        selector: Fitted FeatureSelector for val/test path. If ``None`` and
            ``is_train=True``, a new one is created and fitted.
        is_train: If True, fit preprocessor and selector on ``df``.
            If False, ``preprocessor`` and ``selector`` must be provided.

    Returns:
        Tuple of ``(df_out, feature_names, preprocessor, selector)``.

    Raises:
        ValueError: If ``is_train=False`` and ``preprocessor`` or ``selector``
            is not provided.
    """
    target_col = config.data.target_col

    # ------------------------------------------------------------------
    # Step 1: Preprocessing
    # ------------------------------------------------------------------
    if is_train:
        preprocessor = Preprocessor(config.data)
        df = preprocessor.fit_transform(df)
        logger.info(f"Preprocessing complete: {df.shape[1]} columns")
    else:
        if preprocessor is None:
            raise ValueError(
                "preprocessor must be provided when is_train=False"
            )
        if selector is None:
            raise ValueError(
                "selector must be provided when is_train=False"
            )
        df = preprocessor.transform(df)

    # ------------------------------------------------------------------
    # Step 2: Engineering features (stateless)
    # ------------------------------------------------------------------
    df = build_features(df, config)
    logger.info(f"After engineering: {df.shape[1]} columns")

    # ------------------------------------------------------------------
    # Step 3: Ugly duckling (stateless)
    # ------------------------------------------------------------------
    if config.features.use_ugly_duckling:
        df = build_ugly_duckling_features(df, config)
        logger.info(f"After ugly duckling: {df.shape[1]} columns")

    # ------------------------------------------------------------------
    # Step 4: Feature selection
    # ------------------------------------------------------------------
    if is_train:
        selector = FeatureSelector(config.features)
        selector.fit(df, target_col=target_col)
    else:
        # Transform path: selector already fitted, just record selected cols.
        pass

    logger.info(f"Selected features: {len(selector.selected_cols_)}")  # type: ignore[union-attr]
    return df, selector.selected_cols_, preprocessor, selector  # type: ignore[return-value]
