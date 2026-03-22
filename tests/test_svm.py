"""Tests for SVMBaseline model."""
from __future__ import annotations

import numpy as np

from isic2024.models.svm_baseline import SVMBaseline, _stratified_subsample

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _make_dataset(n: int = 200, n_feat: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    y[:10] = 1
    rng.shuffle(y)
    X = rng.standard_normal((n, n_feat)).astype(np.float32)
    X[y == 1] += 1.0
    split = int(0.8 * n)
    return X[:split], y[:split], X[split:], y[split:]


# ---------------------------------------------------------------------------
# Test 1: predict_proba output in [0, 1]
# ---------------------------------------------------------------------------

def test_svm_predict_proba_range(base_config) -> None:
    X_tr, y_tr, X_val, y_val = _make_dataset()
    model = SVMBaseline(base_config.svm, max_train_samples=200)
    model.fit(X_tr, y_tr, X_val, y_val)
    probs = model.predict_proba(X_val)
    assert probs.shape == (len(X_val),)
    assert probs.min() >= 0.0 and probs.max() <= 1.0


# ---------------------------------------------------------------------------
# Test 2: probabilities are calibrated (sum > 0, not all same)
# ---------------------------------------------------------------------------

def test_svm_probabilities_vary(base_config) -> None:
    X_tr, y_tr, X_val, y_val = _make_dataset()
    model = SVMBaseline(base_config.svm, max_train_samples=200)
    model.fit(X_tr, y_tr, X_val, y_val)
    probs = model.predict_proba(X_val)
    assert probs.std() > 0.0, "All predicted probabilities are identical"


# ---------------------------------------------------------------------------
# Test 3: save / load roundtrip
# ---------------------------------------------------------------------------

def test_svm_save_load(tmp_path, base_config) -> None:
    X_tr, y_tr, X_val, y_val = _make_dataset()
    model = SVMBaseline(base_config.svm, max_train_samples=200)
    model.fit(X_tr, y_tr, X_val, y_val)
    probs_before = model.predict_proba(X_val)

    path = tmp_path / "svm.pkl"
    model.save(path)
    loaded = SVMBaseline.load(path)
    probs_after = loaded.predict_proba(X_val)

    np.testing.assert_allclose(probs_before, probs_after, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test 4: feature_importance returns zeros of correct length
# ---------------------------------------------------------------------------

def test_svm_feature_importance(base_config) -> None:
    X_tr, y_tr, X_val, y_val = _make_dataset(n_feat=10)
    model = SVMBaseline(base_config.svm, max_train_samples=200)
    model.fit(X_tr, y_tr, X_val, y_val)
    imp = model.feature_importance()
    assert imp.shape == (10,)
    assert (imp == 0.0).all()


# ---------------------------------------------------------------------------
# Test 5: subsampling keeps positive rate
# ---------------------------------------------------------------------------

def test_stratified_subsample_preserves_rate() -> None:
    rng = np.random.default_rng(1)
    n = 10_000
    y = (rng.random(n) < 0.02).astype(float)  # 2% positives
    X = rng.standard_normal((n, 5)).astype(np.float32)

    X_sub, y_sub = _stratified_subsample(X, y, max_samples=500, seed=42)
    assert len(X_sub) == 500
    # Positive rate in subsample should be within 1% of original
    orig_rate = y.mean()
    sub_rate = y_sub.mean()
    assert abs(sub_rate - orig_rate) < 0.01, (
        f"Positive rate drifted: orig={orig_rate:.3f} sub={sub_rate:.3f}"
    )


# ---------------------------------------------------------------------------
# Test 6: no subsampling when n < max_samples
# ---------------------------------------------------------------------------

def test_stratified_subsample_no_op_when_small() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 5)).astype(np.float32)
    y = np.zeros(100)
    y[:5] = 1
    X_sub, y_sub = _stratified_subsample(X, y, max_samples=500, seed=0)
    assert len(X_sub) == 100
