"""Rank-based ensemble for combining multiple model predictions.

RankEnsemble converts each model's predicted probabilities to rank percentiles
(0–1), then computes a weighted average. This is robust to probability scale
differences across libraries (LightGBM, XGBoost, CatBoost, SVM produce
uncalibrated scores on different scales).

Ranking preserves the pAUC metric since it's rank-based, and the average of
rank-normalised scores is itself a valid ranking.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


class RankEnsemble:
    """Weighted rank-average of multiple model predictions.

    Example::

        ensemble = RankEnsemble(weights=[1.0, 1.0, 0.5])
        probs = ensemble.predict(predictions_list)

    Args:
        weights: Per-model weights. If None, equal weights are used. Must have
            the same length as the list passed to ``predict``.
    """

    def __init__(self, weights: list[float] | None = None) -> None:
        self._weights = weights

    def predict(self, predictions: list[np.ndarray]) -> np.ndarray:
        """Return rank-averaged predictions.

        Args:
            predictions: List of 1-D arrays, each containing predicted
                positive-class probabilities from one model. All arrays must
                have the same length.

        Returns:
            1-D array of blended scores in [0, 1].

        Raises:
            ValueError: If ``predictions`` is empty or arrays have different
                lengths.
        """
        if not predictions:
            raise ValueError("predictions list is empty")

        n = len(predictions[0])
        for i, p in enumerate(predictions[1:], start=1):
            if len(p) != n:
                raise ValueError(
                    f"predictions[{i}] has length {len(p)}, expected {n}"
                )

        weights = self._weights
        if weights is None:
            weights = [1.0] * len(predictions)
        if len(weights) != len(predictions):
            raise ValueError(
                f"len(weights)={len(weights)} != len(predictions)={len(predictions)}"
            )

        # Convert each model's scores to rank percentiles in [0, 1]
        rank_preds = []
        for pred in predictions:
            ranked = rankdata(pred, method="average")  # ties → average rank
            rank_preds.append(ranked / len(ranked))    # normalise to [0, 1]

        w = np.asarray(weights, dtype=np.float64)
        stacked = np.stack(rank_preds, axis=0)          # (n_models, n_samples)
        blended = np.average(stacked, axis=0, weights=w)
        return blended.astype(np.float64)
