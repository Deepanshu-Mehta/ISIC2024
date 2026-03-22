"""Tests for IsotonicCalibrator, PlattCalibrator, TemperatureScaler, and factory."""
from __future__ import annotations

import numpy as np
import pytest

from isic2024.models.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureScaler,
    calibrator_factory,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_calibration_data(n: int = 500, seed: int = 0):
    """Synthetic over-confident predictions (common GBDT pattern)."""
    rng = np.random.default_rng(seed)
    y_true = (rng.random(n) < 0.1).astype(float)  # 10% positives
    # Predictions shifted toward 0/1 (over-confident)
    y_prob = np.clip(
        np.where(y_true == 1,
                 rng.uniform(0.6, 0.99, n),
                 rng.uniform(0.01, 0.4, n)),
        0.01, 0.99,
    )
    return y_true, y_prob


# ---------------------------------------------------------------------------
# Test 1: IsotonicCalibrator output in [0, 1]
# ---------------------------------------------------------------------------

def test_isotonic_output_range() -> None:
    y_true, y_prob = _make_calibration_data()
    cal = IsotonicCalibrator()
    cal.fit(y_true, y_prob)
    out = cal.transform(y_prob)
    assert out.shape == y_prob.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


# ---------------------------------------------------------------------------
# Test 2: IsotonicCalibrator is monotone (rank-preserving)
# ---------------------------------------------------------------------------

def test_isotonic_is_monotone() -> None:
    """Calibrated scores must be non-decreasing w.r.t. original scores."""
    y_true, y_prob = _make_calibration_data()
    cal = IsotonicCalibrator()
    cal.fit(y_true, y_prob)
    out = cal.transform(y_prob)
    # Sort by original scores; calibrated output must be non-decreasing
    order = np.argsort(y_prob)
    assert np.all(np.diff(out[order]) >= -1e-9), "Isotonic must be non-decreasing"


# ---------------------------------------------------------------------------
# Test 3: PlattCalibrator output in [0, 1]
# ---------------------------------------------------------------------------

def test_platt_output_range() -> None:
    y_true, y_prob = _make_calibration_data()
    cal = PlattCalibrator()
    cal.fit(y_true, y_prob)
    out = cal.transform(y_prob)
    assert out.shape == y_prob.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


# ---------------------------------------------------------------------------
# Test 4: TemperatureScaler with T=1 is approximately identity
# ---------------------------------------------------------------------------

def test_temperature_identity_at_t1() -> None:
    """When temperature=1, transform(p) ≈ p (up to floating-point clipping)."""
    rng = np.random.default_rng(42)
    y_prob = rng.uniform(0.05, 0.95, 200)
    scaler = TemperatureScaler()
    scaler._temperature = 1.0  # force T=1 without fitting
    out = scaler.transform(y_prob)
    np.testing.assert_allclose(out, y_prob, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 5: TemperatureScaler fits without error and T > 0
# ---------------------------------------------------------------------------

def test_temperature_scaler_fits() -> None:
    y_true, y_prob = _make_calibration_data()
    scaler = TemperatureScaler()
    scaler.fit(y_true, y_prob)
    assert scaler.temperature > 0.0
    out = scaler.transform(y_prob)
    assert out.min() >= 0.0 and out.max() <= 1.0


# ---------------------------------------------------------------------------
# Test 6: fit_transform convenience wrapper matches fit then transform
# ---------------------------------------------------------------------------

def test_fit_transform_matches_fit_then_transform() -> None:
    y_true, y_prob = _make_calibration_data(seed=7)

    cal1 = IsotonicCalibrator()
    out1 = cal1.fit_transform(y_true, y_prob)

    cal2 = IsotonicCalibrator()
    cal2.fit(y_true, y_prob)
    out2 = cal2.transform(y_prob)

    np.testing.assert_allclose(out1, out2)


# ---------------------------------------------------------------------------
# Test 7: calibration reduces ECE for over-confident predictions
# ---------------------------------------------------------------------------

def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for k, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (y_prob >= lo) & (y_prob <= hi if k == n_bins - 1 else y_prob < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(y_prob[mask].mean() - y_true[mask].mean())
    return ece


def test_isotonic_reduces_ece() -> None:
    y_true, y_prob = _make_calibration_data(n=2000, seed=99)
    ece_before = _ece(y_true, y_prob)

    cal = IsotonicCalibrator()
    cal.fit(y_true, y_prob)
    ece_after = _ece(y_true, cal.transform(y_prob))

    assert ece_after < ece_before, (
        f"Expected ECE to decrease; before={ece_before:.4f} after={ece_after:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 8: calibrator_factory returns correct types
# ---------------------------------------------------------------------------

def test_factory_returns_isotonic() -> None:
    assert isinstance(calibrator_factory("isotonic"), IsotonicCalibrator)


def test_factory_returns_platt() -> None:
    assert isinstance(calibrator_factory("platt"), PlattCalibrator)


def test_factory_returns_temperature() -> None:
    assert isinstance(calibrator_factory("temperature"), TemperatureScaler)


def test_factory_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown calibration method"):
        calibrator_factory("nonexistent")
