"""Tests for RankEnsemble."""
from __future__ import annotations

import numpy as np
import pytest

from isic2024.models.ensemble import RankEnsemble


# ---------------------------------------------------------------------------
# Test 1: output is in [0, 1]
# ---------------------------------------------------------------------------

def test_rank_ensemble_output_range() -> None:
    rng = np.random.default_rng(0)
    preds = [rng.random(100) for _ in range(3)]
    ensemble = RankEnsemble()
    out = ensemble.predict(preds)
    assert out.shape == (100,)
    assert out.min() >= 0.0 and out.max() <= 1.0


# ---------------------------------------------------------------------------
# Test 2: equal weights produce same result as no weights
# ---------------------------------------------------------------------------

def test_rank_ensemble_equal_weights() -> None:
    rng = np.random.default_rng(1)
    preds = [rng.random(50) for _ in range(3)]
    out_default = RankEnsemble().predict(preds)
    out_explicit = RankEnsemble(weights=[1.0, 1.0, 1.0]).predict(preds)
    np.testing.assert_allclose(out_default, out_explicit)


# ---------------------------------------------------------------------------
# Test 3: single model returns its own rank-normalised scores
# ---------------------------------------------------------------------------

def test_rank_ensemble_single_model() -> None:
    preds = [np.array([0.1, 0.9, 0.5, 0.3])]
    out = RankEnsemble().predict(preds)
    # Ranks: 0.1→1, 0.3→2, 0.5→3, 0.9→4 → normalised by 4
    expected = np.array([1, 4, 3, 2]) / 4.0
    np.testing.assert_allclose(out, expected)


# ---------------------------------------------------------------------------
# Test 4: ranking is preserved (monotone w.r.t. original order)
# ---------------------------------------------------------------------------

def test_rank_ensemble_preserves_order() -> None:
    # Two models agree on ordering → blended output should preserve it
    preds = [
        np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
        np.array([0.2, 0.4, 0.6, 0.8, 1.0]),
    ]
    out = RankEnsemble().predict(preds)
    assert np.all(np.diff(out) > 0), "Output should be strictly increasing"


# ---------------------------------------------------------------------------
# Test 5: weighted blend gives more weight to first model
# ---------------------------------------------------------------------------

def test_rank_ensemble_weights_effect() -> None:
    rng = np.random.default_rng(2)
    p1 = rng.random(200)
    # p2 is negatively correlated with p1
    p2 = 1.0 - p1 + rng.normal(0, 0.05, 200)
    p2 = np.clip(p2, 0, 1)

    out_equal = RankEnsemble(weights=[1.0, 1.0]).predict([p1, p2])
    out_weighted = RankEnsemble(weights=[10.0, 1.0]).predict([p1, p2])

    # Rank correlation of out_weighted with p1 should be higher than out_equal
    from scipy.stats import spearmanr
    r_equal = spearmanr(out_equal, p1).statistic
    r_weighted = spearmanr(out_weighted, p1).statistic
    assert r_weighted > r_equal


# ---------------------------------------------------------------------------
# Test 6: empty predictions list raises ValueError
# ---------------------------------------------------------------------------

def test_rank_ensemble_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        RankEnsemble().predict([])


# ---------------------------------------------------------------------------
# Test 7: mismatched lengths raise ValueError
# ---------------------------------------------------------------------------

def test_rank_ensemble_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        RankEnsemble().predict([np.ones(10), np.ones(20)])


# ---------------------------------------------------------------------------
# Test 8: mismatched weights length raises ValueError
# ---------------------------------------------------------------------------

def test_rank_ensemble_weight_mismatch_raises() -> None:
    preds = [np.ones(10), np.ones(10)]
    with pytest.raises(ValueError):
        RankEnsemble(weights=[1.0]).predict(preds)


# ---------------------------------------------------------------------------
# Test 9: ties handled gracefully (no NaN in output)
# ---------------------------------------------------------------------------

def test_rank_ensemble_handles_ties() -> None:
    preds = [np.ones(20) * 0.5, np.ones(20) * 0.5]
    out = RankEnsemble().predict(preds)
    assert not np.any(np.isnan(out))
    assert not np.any(np.isinf(out))
