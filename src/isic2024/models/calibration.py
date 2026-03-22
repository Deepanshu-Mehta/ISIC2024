"""Probability calibration wrappers.

Three calibration methods are provided:

IsotonicCalibrator (PRIMARY for GBDT)
    Non-parametric isotonic regression. Handles non-linear miscalibration.
    GBDT already optimises log-loss so isotonic handles the residual best.

PlattCalibrator
    Logistic regression fit on predicted probabilities. Good when
    miscalibration is approximately sigmoidal (e.g., SVMs).

TemperatureScaler
    Divides logits by a learnable scalar T, then applies sigmoid.
    Primarily designed for neural-network logits; included for completeness.

All three share the same interface::

    cal.fit(y_true, y_prob)          # fit on OOF predictions
    calibrated = cal.transform(y_prob)  # apply to new predictions

Calibration preserves ranking → does NOT affect pAUC. It improves ECE/Brier.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseCalibrator(ABC):
    """Common interface for all calibration wrappers."""

    @abstractmethod
    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> BaseCalibrator:
        """Fit the calibrator on held-out predictions."""

    @abstractmethod
    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities in [0, 1]."""

    def fit_transform(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> np.ndarray:
        """Fit then transform (convenience wrapper)."""
        return self.fit(y_true, y_prob).transform(y_prob)


# ---------------------------------------------------------------------------
# Isotonic Regression (primary)
# ---------------------------------------------------------------------------

class IsotonicCalibrator(BaseCalibrator):
    """Non-parametric monotone calibration via isotonic regression.

    Preferred for GBDT outputs because miscalibration is often non-linear.
    ``out_of_bounds='clip'`` ensures predictions outside the training range
    are clipped to [0, 1] rather than extrapolated.
    """

    def __init__(self) -> None:
        self._iso = IsotonicRegression(out_of_bounds="clip")

    def fit(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> IsotonicCalibrator:
        """Fit isotonic regression on OOF predictions.

        Args:
            y_true: Ground-truth binary labels.
            y_prob: Predicted positive-class probabilities.

        Returns:
            self
        """
        self._iso.fit(
            np.asarray(y_prob, dtype=np.float64),
            np.asarray(y_true, dtype=np.float64),
        )
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply isotonic mapping and clip to [0, 1].

        Args:
            y_prob: Predicted positive-class probabilities.

        Returns:
            Calibrated probabilities in [0, 1].
        """
        calibrated = self._iso.predict(np.asarray(y_prob, dtype=np.float64))
        return np.clip(calibrated, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Platt Calibration (logistic regression)
# ---------------------------------------------------------------------------

class PlattCalibrator(BaseCalibrator):
    """Sigmoidal calibration via logistic regression (Platt scaling).

    Fits a logistic regression on predicted probabilities. Good for SVMs
    and when miscalibration is approximately sigmoidal.
    """

    def __init__(self) -> None:
        self._lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> PlattCalibrator:
        """Fit logistic regression on OOF predictions.

        Args:
            y_true: Ground-truth binary labels.
            y_prob: Predicted positive-class probabilities.

        Returns:
            self
        """
        X = np.asarray(y_prob, dtype=np.float64).reshape(-1, 1)
        self._lr.fit(X, np.asarray(y_true, dtype=np.float64))
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply Platt mapping.

        Args:
            y_prob: Predicted positive-class probabilities.

        Returns:
            Calibrated probabilities in [0, 1].
        """
        X = np.asarray(y_prob, dtype=np.float64).reshape(-1, 1)
        return self._lr.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Temperature Scaling
# ---------------------------------------------------------------------------

class TemperatureScaler(BaseCalibrator):
    """Scalar temperature calibration: sigmoid(logit(p) / T).

    Primarily designed for neural-network logits. Included here for
    comparison. Finds T by minimising negative log-likelihood on the
    calibration set via a grid search over [0.1, 10.0].

    T > 1 → softer (more uncertain) predictions.
    T < 1 → sharper (more confident) predictions.
    T = 1 → identity transform.
    """

    def __init__(self) -> None:
        self._temperature: float = 1.0

    @property
    def temperature(self) -> float:
        """Fitted temperature value."""
        return self._temperature

    def fit(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> TemperatureScaler:
        """Find T that minimises NLL on calibration data.

        Args:
            y_true: Ground-truth binary labels.
            y_prob: Predicted positive-class probabilities.

        Returns:
            self
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-7, 1 - 1e-7)
        logits = np.log(y_prob / (1.0 - y_prob))  # inverse sigmoid

        best_t = 1.0
        best_nll = np.inf
        for t in np.linspace(0.1, 10.0, 500):
            scaled = self._sigmoid(logits / t)
            nll = -np.mean(
                y_true * np.log(scaled + 1e-15)
                + (1.0 - y_true) * np.log(1.0 - scaled + 1e-15)
            )
            if nll < best_nll:
                best_nll = nll
                best_t = float(t)

        self._temperature = best_t
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply temperature scaling.

        Args:
            y_prob: Predicted positive-class probabilities.

        Returns:
            Calibrated probabilities in [0, 1].
        """
        y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-7, 1 - 1e-7)
        logits = np.log(y_prob / (1.0 - y_prob))
        return self._sigmoid(logits / self._temperature)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_CALIBRATOR_REGISTRY: dict[str, type[BaseCalibrator]] = {
    "isotonic": IsotonicCalibrator,
    "platt": PlattCalibrator,
    "temperature": TemperatureScaler,
}


def calibrator_factory(method: str) -> BaseCalibrator:
    """Return a calibrator instance by name.

    Args:
        method: One of ``"isotonic"``, ``"platt"``, ``"temperature"``.

    Returns:
        Unfitted calibrator instance.

    Raises:
        ValueError: If ``method`` is not registered.
    """
    method = method.lower()
    if method not in _CALIBRATOR_REGISTRY:
        raise ValueError(
            f"Unknown calibration method '{method}'. "
            f"Choose from: {list(_CALIBRATOR_REGISTRY)}"
        )
    return _CALIBRATOR_REGISTRY[method]()
