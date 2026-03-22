"""SVM baseline model — classical probabilistic model for course requirements.

Uses SVC with RBF kernel + Platt Scaling (probability=True), wrapped with
StandardScaler. This satisfies the "Advanced ML" course requirement for a
classical probabilistic model alongside the GBDT ensemble.

⚠ Speed warning: SVC with probability=True on 401K rows takes hours.
  Always subsample before training (see SVMBaseline.fit docstring).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from isic2024.models.gbdt import BaseModel


class SVMBaseline(BaseModel):
    """SVM with RBF kernel + Platt Scaling, scaled with StandardScaler.

    Satisfies course requirement for a classical probabilistic model.
    Subsamples to ``max_train_samples`` rows to keep training tractable.
    """

    def __init__(self, config: object, max_train_samples: int = 20_000) -> None:
        self._config = config
        self._max_train_samples = max_train_samples
        self._scaler = StandardScaler()
        self._model: SVC | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        seed: int = 42,
    ) -> SVMBaseline:
        """Fit on a stratified subsample of training data.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features (used for logging only — SVM has no early stopping).
            y_val: Validation labels (used for logging only).
            seed: Random seed for subsampling.

        Returns:
            self
        """
        cfg = self._config

        # Stratified subsample to keep SVC tractable
        X_sub, y_sub = _stratified_subsample(
            X_train, y_train, self._max_train_samples, seed
        )
        n_pos = int(y_sub.sum())
        logger.info(
            f"SVMBaseline: training on {len(X_sub):,} rows "
            f"({n_pos} positive, {len(X_sub) - n_pos} negative)"
        )

        X_sub_scaled = self._scaler.fit_transform(X_sub)

        self._model = SVC(
            kernel=cfg.kernel,
            C=cfg.C,
            gamma=cfg.gamma,
            probability=cfg.probability,  # Platt Scaling
            class_weight=cfg.class_weight,
            random_state=seed,
        )
        self._model.fit(X_sub_scaled, y_sub)

        # Log validation AUC for comparison
        from sklearn.metrics import roc_auc_score

        X_val_scaled = self._scaler.transform(X_val)
        val_probs = self._model.predict_proba(X_val_scaled)[:, 1]
        if len(np.unique(y_val)) > 1:
            val_auc = roc_auc_score(y_val, val_probs)
            logger.info(f"SVMBaseline: val_AUC={val_auc:.4f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities for the positive class."""
        X_scaled = self._scaler.transform(X)
        return self._model.predict_proba(X_scaled)[:, 1]

    def feature_importance(self) -> np.ndarray:
        """SVM does not have feature importances — returns zeros."""
        if self._model is None:
            return np.array([])
        n = self._model.n_features_in_
        logger.warning("SVMBaseline.feature_importance(): SVM has no importances; returning zeros")
        return np.zeros(n, dtype=np.float64)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved SVMBaseline → {path}")

    @classmethod
    def load(cls, path: str | Path) -> SVMBaseline:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Loaded SVMBaseline ← {path}")
        return obj


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _stratified_subsample(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a stratified subsample of at most ``max_samples`` rows."""
    if len(X) <= max_samples:
        return X, y

    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    # Preserve original positive rate
    pos_rate = len(pos_idx) / len(y)
    n_pos = max(1, int(max_samples * pos_rate))
    n_neg = max_samples - n_pos

    chosen_pos = rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False)
    chosen_neg = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
    idx = np.concatenate([chosen_pos, chosen_neg])
    rng.shuffle(idx)

    return X[idx], y[idx]
