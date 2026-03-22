"""Evaluation metrics for the ISIC 2024 competition.

The primary metric is pAUC (partial AUC above 80% TPR). The exact implementation
uses label-flipping + prediction negation, matching the official Kaggle scoring
notebook. Do NOT use sklearn's roc_auc_score(max_fpr=0.2) — it applies McClish
normalization and gives different numbers.

Calibration metrics (ECE, Brier) do not affect pAUC (which is rank-based) but
are useful for clinical interpretation and comparing calibration methods.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import auc, brier_score_loss, roc_auc_score, roc_curve


def compute_pauc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_tpr: float = 0.80,
) -> float:
    """Compute partial AUC above ``min_tpr`` TPR (exact Kaggle competition metric).

    Implements the official scoring formula via label-flipping and prediction
    negation. This is NOT equivalent to ``sklearn.metrics.roc_auc_score`` with
    ``max_fpr=0.2`` — sklearn applies McClish normalization which gives different
    values.

    The maximum possible score is ``1 - min_tpr`` (default 0.20).

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_pred: Predicted probabilities in [0, 1].
        min_tpr: Minimum TPR threshold. Default 0.80 (competition standard).

    Returns:
        pAUC score in [0.0, 1 - min_tpr]. Returns 0.0 for degenerate inputs.
    """
    v_gt = abs(np.asarray(y_true, dtype=np.float64) - 1)  # flip labels
    v_pred = -1.0 * np.asarray(y_pred, dtype=np.float64)  # negate predictions
    max_fpr = abs(1.0 - min_tpr)  # 0.20 for min_tpr=0.80

    fpr, tpr, _ = roc_curve(v_gt, v_pred, drop_intermediate=True)

    # Edge case: too few points or curve doesn't reach max_fpr
    if len(fpr) < 2:
        return 0.0
    if fpr[-1] <= max_fpr:
        return float(auc(fpr, tpr))

    stop = np.searchsorted(fpr, max_fpr, "right")
    if stop == 0:
        return 0.0

    # Interpolate TPR at the max_fpr boundary
    tpr_at_boundary = float(
        np.interp(max_fpr, [fpr[stop - 1], fpr[stop]], [tpr[stop - 1], tpr[stop]])
    )

    fpr_clipped = np.append(fpr[:stop], max_fpr)
    tpr_clipped = np.append(tpr[:stop], tpr_at_boundary)
    return float(auc(fpr_clipped, tpr_clipped))


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Measures the average gap between predicted confidence and observed accuracy,
    weighted by bin size. Lower is better; perfectly calibrated → ECE = 0.

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_prob: Predicted probabilities in [0, 1].
        n_bins: Number of equal-width probability bins. Default 10.

    Returns:
        ECE in [0.0, 1.0].
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n = len(y_true)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for k, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        # Include upper bound for the last bin so predictions at exactly 1.0 are captured
        if k == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        bin_size = mask.sum()
        mean_conf = y_prob[mask].mean()
        frac_pos = y_true[mask].mean()
        ece += (bin_size / n) * abs(mean_conf - frac_pos)

    return float(ece)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute all evaluation metrics in one call.

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_pred: Predicted probabilities in [0, 1].

    Returns:
        Dict with keys ``pauc``, ``roc_auc``, ``brier``, ``ece``.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return {
        "pauc": compute_pauc(y_true, y_pred),
        "roc_auc": float(roc_auc_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_pred)),
        "ece": compute_ece(y_true, y_pred),
    }
