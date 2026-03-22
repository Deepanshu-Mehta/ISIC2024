"""Tests for GBDT model wrappers (LightGBM, XGBoost, CatBoost) and model_factory."""
from __future__ import annotations

import numpy as np
import pytest

from isic2024.models.gbdt import (
    CatBoostWrapper,
    LGBMWrapper,
    XGBWrapper,
    model_factory,
)

# ---------------------------------------------------------------------------
# Synthetic dataset fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_dataset():
    """300-row imbalanced dataset: 10 positives, 20 features."""
    rng = np.random.default_rng(0)
    n, n_feat = 300, 20
    y = np.zeros(n)
    y[:10] = 1
    rng.shuffle(y)
    X = rng.standard_normal((n, n_feat))
    # Make positives slightly separable
    X[y == 1] += 0.5
    split = int(0.8 * n)
    return (
        X[:split].astype(np.float32),
        y[:split],
        X[split:].astype(np.float32),
        y[split:],
    )


@pytest.fixture()
def lgbm_config(base_config):
    return base_config.lgbm


@pytest.fixture()
def xgb_config(base_config):
    return base_config.xgb


@pytest.fixture()
def catboost_config(base_config):
    return base_config.catboost


# ---------------------------------------------------------------------------
# Test 1: predict_proba output in [0, 1]
# ---------------------------------------------------------------------------

def test_lgbm_predict_proba_range(small_dataset, lgbm_config) -> None:
    X_tr, y_tr, X_val, y_val = small_dataset
    model = LGBMWrapper(lgbm_config)
    model.fit(X_tr, y_tr, X_val, y_val)
    probs = model.predict_proba(X_val)
    assert probs.shape == (len(X_val),)
    assert probs.min() >= 0.0 and probs.max() <= 1.0


def test_xgb_predict_proba_range(small_dataset, xgb_config) -> None:
    X_tr, y_tr, X_val, y_val = small_dataset
    model = XGBWrapper(xgb_config)
    model.fit(X_tr, y_tr, X_val, y_val)
    probs = model.predict_proba(X_val)
    assert probs.shape == (len(X_val),)
    assert probs.min() >= 0.0 and probs.max() <= 1.0


def test_catboost_predict_proba_range(small_dataset, catboost_config) -> None:
    X_tr, y_tr, X_val, y_val = small_dataset
    model = CatBoostWrapper(catboost_config)
    model.fit(X_tr, y_tr, X_val, y_val)
    probs = model.predict_proba(X_val)
    assert probs.shape == (len(X_val),)
    assert probs.min() >= 0.0 and probs.max() <= 1.0


# ---------------------------------------------------------------------------
# Test 2: feature_importance length matches n_features
# ---------------------------------------------------------------------------

def test_lgbm_feature_importance_length(small_dataset, lgbm_config) -> None:
    X_tr, y_tr, X_val, y_val = small_dataset
    model = LGBMWrapper(lgbm_config)
    model.fit(X_tr, y_tr, X_val, y_val)
    imp = model.feature_importance()
    assert imp.shape == (X_tr.shape[1],)


def test_xgb_feature_importance_length(small_dataset, xgb_config) -> None:
    X_tr, y_tr, X_val, y_val = small_dataset
    model = XGBWrapper(xgb_config)
    model.fit(X_tr, y_tr, X_val, y_val)
    imp = model.feature_importance()
    assert imp.shape == (X_tr.shape[1],)


def test_catboost_feature_importance_length(small_dataset, catboost_config) -> None:
    X_tr, y_tr, X_val, y_val = small_dataset
    model = CatBoostWrapper(catboost_config)
    model.fit(X_tr, y_tr, X_val, y_val)
    imp = model.feature_importance()
    assert imp.shape == (X_tr.shape[1],)


# ---------------------------------------------------------------------------
# Test 3: save / load roundtrip
# ---------------------------------------------------------------------------

def test_lgbm_save_load(tmp_path, small_dataset, lgbm_config) -> None:
    X_tr, y_tr, X_val, y_val = small_dataset
    model = LGBMWrapper(lgbm_config)
    model.fit(X_tr, y_tr, X_val, y_val)
    probs_before = model.predict_proba(X_val)

    save_path = tmp_path / "lgbm.pkl"
    model.save(save_path)
    loaded = LGBMWrapper.load(save_path)
    probs_after = loaded.predict_proba(X_val)

    np.testing.assert_allclose(probs_before, probs_after, rtol=1e-5)


def test_xgb_save_load(tmp_path, small_dataset, xgb_config) -> None:
    X_tr, y_tr, X_val, y_val = small_dataset
    model = XGBWrapper(xgb_config)
    model.fit(X_tr, y_tr, X_val, y_val)
    probs_before = model.predict_proba(X_val)

    save_path = tmp_path / "xgb.pkl"
    model.save(save_path)
    loaded = XGBWrapper.load(save_path)
    probs_after = loaded.predict_proba(X_val)

    np.testing.assert_allclose(probs_before, probs_after, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test 4: model_factory returns correct type
# ---------------------------------------------------------------------------

def test_model_factory_lgbm(lgbm_config) -> None:
    model = model_factory("lgbm", lgbm_config)
    assert isinstance(model, LGBMWrapper)


def test_model_factory_xgb(xgb_config) -> None:
    model = model_factory("xgb", xgb_config)
    assert isinstance(model, XGBWrapper)


def test_model_factory_catboost(catboost_config) -> None:
    model = model_factory("catboost", catboost_config)
    assert isinstance(model, CatBoostWrapper)


def test_model_factory_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        model_factory("unknown_model", object())
