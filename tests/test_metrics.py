"""Tests for evaluation metrics: pAUC, ECE, compute_metrics."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from isic2024.evaluation.metrics import compute_ece, compute_metrics, compute_pauc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_realistic(n: int = 500, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Realistic imbalanced binary data with 2% positives."""
    rng = np.random.default_rng(seed)
    y_true = (rng.random(n) < 0.02).astype(int)
    # Scores correlated with label but noisy
    y_pred = np.clip(y_true * 0.6 + rng.random(n) * 0.4, 0, 1)
    return y_true.astype(float), y_pred


# ---------------------------------------------------------------------------
# Test 1: perfect classifier → pAUC = 0.20
# ---------------------------------------------------------------------------

def test_perfect_classifier() -> None:
    y_true = np.array([1] * 20 + [0] * 180, dtype=float)
    # Perfect scores: all positives score 1.0, all negatives score 0.0
    y_pred = np.array([1.0] * 20 + [0.0] * 180)
    result = compute_pauc(y_true, y_pred)
    assert abs(result - 0.20) < 1e-6, f"Perfect classifier pAUC={result:.6f}, expected 0.20"


# ---------------------------------------------------------------------------
# Test 2: random classifier → pAUC ≈ 0.02
# ---------------------------------------------------------------------------

def test_random_classifier() -> None:
    rng = np.random.default_rng(42)
    n = 10_000
    y_true = (rng.random(n) < 0.1).astype(float)
    y_pred = rng.random(n)
    result = compute_pauc(y_true, y_pred)
    # Random pAUC ≈ 0.5 × max_fpr² = 0.02 (triangle under diagonal)
    assert 0.01 < result < 0.05, f"Random pAUC={result:.4f}, expected ≈0.02"


# ---------------------------------------------------------------------------
# Test 3: all-same prediction → no crash, returns random-baseline pAUC (≈0.02)
# ---------------------------------------------------------------------------

def test_all_same_prediction() -> None:
    y_true = np.array([1, 0, 1, 0, 0], dtype=float)
    y_pred = np.full(5, 0.5)
    result = compute_pauc(y_true, y_pred)
    # All-same scores → diagonal ROC (random) → pAUC ≈ 0.02 (not 0.0); must not crash
    assert 0.0 <= result <= 0.021


# ---------------------------------------------------------------------------
# Test 4: differs from sklearn roc_auc_score(max_fpr=0.2)
# ---------------------------------------------------------------------------

def test_differs_from_sklearn_max_fpr() -> None:
    y_true, y_pred = _make_realistic(n=1000, seed=7)
    # Ensure we actually have both classes
    if y_true.sum() == 0:
        y_true[0] = 1.0

    our_pauc = compute_pauc(y_true, y_pred)
    sklearn_pauc = roc_auc_score(y_true, y_pred, max_fpr=0.2)

    # sklearn applies McClish normalization → different value
    assert our_pauc != pytest.approx(sklearn_pauc, rel=1e-3), (
        f"pAUC={our_pauc:.6f} matches sklearn={sklearn_pauc:.6f} — "
        "should differ due to McClish normalization"
    )


# ---------------------------------------------------------------------------
# Test 5: ECE ≈ 0 for perfectly calibrated predictions
# ---------------------------------------------------------------------------

def test_ece_perfect_calibration() -> None:
    # Each bin: predicted probability ≈ observed fraction positive
    rng = np.random.default_rng(0)
    n = 10_000
    y_prob = rng.uniform(0, 1, n)
    # Sample labels according to predicted probability (perfectly calibrated)
    y_true = (rng.random(n) < y_prob).astype(float)
    ece = compute_ece(y_true, y_prob, n_bins=10)
    assert ece < 0.05, f"ECE={ece:.4f}, expected < 0.05 for calibrated predictions"


# ---------------------------------------------------------------------------
# Test 6: ECE ≈ 0.5 for worst-case miscalibration
# ---------------------------------------------------------------------------

def test_ece_worst_calibration() -> None:
    # All predictions = 1.0 but 50% are actually 0
    n = 1000
    y_true = np.array([1, 0] * (n // 2), dtype=float)
    y_pred = np.ones(n)
    ece = compute_ece(y_true, y_pred, n_bins=10)
    # All weight in the last bin; |confidence - fraction_pos| = |1.0 - 0.5| = 0.5
    assert abs(ece - 0.5) < 0.01, f"ECE={ece:.4f}, expected ≈ 0.50"


# ---------------------------------------------------------------------------
# Test 7: compute_metrics returns dict with all keys in [0, 1]
# ---------------------------------------------------------------------------

def test_compute_metrics_returns_dict() -> None:
    y_true, y_pred = _make_realistic(n=500, seed=1)
    if y_true.sum() == 0:
        y_true[0] = 1.0

    metrics = compute_metrics(y_true, y_pred)

    assert set(metrics.keys()) == {"pauc", "roc_auc", "brier", "ece"}
    for key, val in metrics.items():
        assert 0.0 <= val <= 1.0, f"metrics[{key!r}] = {val:.4f} outside [0, 1]"
