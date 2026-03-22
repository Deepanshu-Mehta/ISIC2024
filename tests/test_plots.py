"""Tests for evaluation/plots.py.

All tests use the Agg backend (no display required) and verify that each
plotting function returns a matplotlib Figure without errors.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from isic2024.evaluation.plots import (
    plot_feature_importance,
    plot_reliability_diagram,
    plot_roc_curves,
    plot_score_distributions,
)


@pytest.fixture()
def synthetic_binary():
    """200-sample binary classification data with predicted scores."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_pred_a = rng.uniform(0, 1, size=200)
    y_pred_b = np.clip(y_true + rng.normal(0, 0.3, size=200), 0, 1)
    return y_true, {"ModelA": y_pred_a, "ModelB": y_pred_b}


def test_plot_roc_curves_returns_figure(synthetic_binary):
    y_true, oof_dict = synthetic_binary
    fig = plot_roc_curves(y_true, oof_dict)
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_score_distributions_returns_figure(synthetic_binary):
    y_true, oof_dict = synthetic_binary
    fig = plot_score_distributions(y_true, oof_dict)
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_reliability_diagram_returns_figure(synthetic_binary):
    y_true, oof_dict = synthetic_binary
    y_prob = oof_dict["ModelB"]
    fig = plot_reliability_diagram(y_true, y_prob, n_bins=10)
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_feature_importance_returns_figure():
    rng = np.random.default_rng(1)
    names = [f"feat_{i}" for i in range(50)]
    imps = rng.uniform(0, 1, size=50)
    fig = plot_feature_importance(names, imps, top_n=20)
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)
