"""GBDT model wrappers: LightGBM, XGBoost, CatBoost.

Each wrapper follows the same interface:
    model.fit(X_train, y_train, X_val, y_val)
    probs = model.predict_proba(X)          # 1-D array of positive-class probs
    imp = model.feature_importance()        # 1-D array aligned with feature columns
    model.save(path) / model.load(path)

All wrappers use sklearn-compatible estimators for consistency:
    - LGBMClassifier, XGBClassifier, CatBoostClassifier
    - scale_pos_weight / class_weights to handle extreme imbalance (~1020:1)
    - Early stopping on validation AUC
    - Seed 42 for reproducibility (overridable per-call for seed averaging)
"""
from __future__ import annotations

import math
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseModel(ABC):
    """Common interface for all model wrappers."""

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        seed: int = 42,
    ) -> BaseModel:
        """Fit the model on training data with early stopping on validation."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probabilities for the positive class."""

    @abstractmethod
    def feature_importance(self) -> np.ndarray:
        """Return feature importances aligned with training feature columns."""

    def save(self, path: str | Path) -> None:
        """Pickle the fitted model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved {self.__class__.__name__} → {path}")

    @classmethod
    def load(cls, path: str | Path) -> BaseModel:
        """Load a pickled model from disk."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Loaded {obj.__class__.__name__} ← {path}")
        return obj


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

class LGBMWrapper(BaseModel):
    """LightGBM binary classifier wrapper (sklearn API)."""

    def __init__(self, config: object) -> None:
        self._config = config
        self._model = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        seed: int = 42,
    ) -> LGBMWrapper:
        import lightgbm as lgb
        from lightgbm import LGBMClassifier

        cfg = self._config
        self._model = LGBMClassifier(
            objective=cfg.objective,
            metric=cfg.metric,
            verbosity=cfg.verbosity,
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves,
            max_depth=cfg.max_depth,
            min_child_samples=cfg.min_child_samples,
            # Weight-based leaf constraint (equiv. to XGB min_child_weight).
            # With scale_pos_weight=100, 1 positive sample contributes ~50 hessian
            # weight → satisfies min_child_weight=10, enabling fine-grained splits.
            min_child_weight=cfg.min_sum_hessian_in_leaf,
            feature_fraction=cfg.feature_fraction,
            bagging_fraction=cfg.bagging_fraction,
            bagging_freq=cfg.bagging_freq,
            reg_alpha=cfg.lambda_l1,
            reg_lambda=cfg.lambda_l2,
            # scale_pos_weight handles extreme imbalance; NOT is_unbalance
            # (is_unbalance distorts predicted probabilities)
            scale_pos_weight=cfg.scale_pos_weight,
            random_state=seed,
        )
        self._model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(-1),  # suppress per-round output
            ],
        )
        best_iter = self._model.best_iteration_
        val_auc = self._model.best_score_["valid_0"][cfg.metric]
        logger.info(
            f"LGBMWrapper: best_iteration={best_iter}  val_{cfg.metric}={val_auc:.4f}"
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    def feature_importance(self) -> np.ndarray:
        return self._model.feature_importances_.astype(np.float64)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

class XGBWrapper(BaseModel):
    """XGBoost binary classifier wrapper (sklearn API)."""

    def __init__(self, config: object) -> None:
        self._config = config
        self._model = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        seed: int = 42,
    ) -> XGBWrapper:
        from xgboost import XGBClassifier

        cfg = self._config
        self._model = XGBClassifier(
            objective=cfg.objective,
            eval_metric=cfg.eval_metric,
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            min_child_weight=cfg.min_child_weight,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            # max_delta_step=1 stabilises gradient updates under extreme imbalance
            max_delta_step=cfg.max_delta_step,
            scale_pos_weight=cfg.scale_pos_weight,
            verbosity=cfg.verbosity,
            early_stopping_rounds=cfg.early_stopping_rounds,
            random_state=seed,
        )
        self._model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        best_iter = self._model.best_iteration
        val_auc = self._model.best_score
        logger.info(
            f"XGBWrapper: best_iteration={best_iter}  val_{cfg.eval_metric}={val_auc:.4f}"
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    def feature_importance(self) -> np.ndarray:
        return self._model.feature_importances_.astype(np.float64)


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------

class CatBoostWrapper(BaseModel):
    """CatBoost binary classifier wrapper (sklearn API)."""

    def __init__(self, config: object) -> None:
        self._config = config
        self._model = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        seed: int = 42,
    ) -> CatBoostWrapper:
        from catboost import CatBoostClassifier

        cfg = self._config
        n_pos = int(y_train.sum())
        n_neg = int(len(y_train) - n_pos)

        # Try SqrtBalanced first; fall back to manual class_weights
        try:
            model = CatBoostClassifier(
                iterations=cfg.iterations,
                learning_rate=cfg.learning_rate,
                depth=cfg.depth,
                l2_leaf_reg=cfg.l2_leaf_reg,
                early_stopping_rounds=cfg.early_stopping_rounds,
                verbose=cfg.verbose,
                random_seed=seed,
                eval_metric="AUC",
                auto_class_weights="SqrtBalanced",
            )
            model.fit(X_train, y_train, eval_set=(X_val, y_val))
        except (TypeError, ValueError):
            # auto_class_weights not available in this version → manual weights
            sqrt_ratio = math.sqrt(n_neg / max(n_pos, 1))
            logger.warning(
                f"CatBoost: auto_class_weights unavailable; "
                f"using class_weights={{0: 1, 1: {sqrt_ratio:.1f}}}"
            )
            model = CatBoostClassifier(
                iterations=cfg.iterations,
                learning_rate=cfg.learning_rate,
                depth=cfg.depth,
                l2_leaf_reg=cfg.l2_leaf_reg,
                early_stopping_rounds=cfg.early_stopping_rounds,
                verbose=cfg.verbose,
                random_seed=seed,
                eval_metric="AUC",
                class_weights={0: 1, 1: sqrt_ratio},
            )
            model.fit(X_train, y_train, eval_set=(X_val, y_val))

        self._model = model
        logger.info(
            f"CatBoostWrapper: best_iteration={model.best_iteration_}  "
            f"val_AUC={model.best_score_['validation']['AUC']:.4f}"
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    def feature_importance(self) -> np.ndarray:
        return self._model.get_feature_importance().astype(np.float64)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[BaseModel]] = {
    "lgbm": LGBMWrapper,
    "xgb": XGBWrapper,
    "catboost": CatBoostWrapper,
}


def model_factory(name: str, config: object) -> BaseModel:
    """Return a model wrapper by name.

    Args:
        name: One of ``"lgbm"``, ``"xgb"``, ``"catboost"``.
        config: Corresponding *Config dataclass instance.

    Returns:
        Unfitted model wrapper.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(_REGISTRY)}")
    return _REGISTRY[name](config)
