"""Visualization utilities for ISIC 2024 evaluation.

All functions return a matplotlib Figure and accept an optional ``ax`` parameter
for embedding into larger subplot grids.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_curve, auc


def plot_roc_curves(
    y_true: np.ndarray,
    oof_dict: dict[str, np.ndarray],
    ax: Any = None,
) -> Any:
    """Plot ROC curves for multiple models with a vertical line at FPR=0.20.

    Args:
        y_true: Binary ground-truth labels.
        oof_dict: Mapping of model name → predicted probabilities.
        ax: Optional matplotlib Axes. If None, a new Figure + Axes is created.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.get_figure()

    for name, y_pred in oof_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", linewidth=1.5)

    ax.axvline(x=0.20, color="gray", linestyle="--", linewidth=1, label="FPR=0.20 (pAUC boundary)")
    ax.plot([0, 1], [0, 1], color="lightgray", linestyle=":", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.01])

    return fig


def plot_score_distributions(
    y_true: np.ndarray,
    oof_dict: dict[str, np.ndarray],
    ax: Any = None,
) -> Any:
    """Plot overlapping score histograms for benign vs malignant per model.

    Args:
        y_true: Binary ground-truth labels.
        oof_dict: Mapping of model name → predicted probabilities.
        ax: Optional matplotlib Axes. If None, a new Figure + Axes is created.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    n_models = len(oof_dict)
    fig = None
    if ax is None:
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), squeeze=False)
        axes = axes[0]
    else:
        # Single ax provided — plot all models overlapping
        axes = [ax] * n_models
        fig = ax.get_figure()

    y_true = np.asarray(y_true)
    for i, (name, y_pred) in enumerate(oof_dict.items()):
        y_pred = np.asarray(y_pred)
        cur_ax = axes[i]
        cur_ax.hist(y_pred[y_true == 0], bins=50, alpha=0.6, label="Benign", density=True, color="steelblue")
        cur_ax.hist(y_pred[y_true == 1], bins=50, alpha=0.8, label="Malignant", density=True, color="tomato")
        cur_ax.set_xlabel("Predicted Probability")
        cur_ax.set_ylabel("Density")
        cur_ax.set_title(f"Score Distribution — {name}")
        cur_ax.legend(fontsize=8)

    if fig is None:
        fig = axes[0].get_figure()
    return fig


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    ax: Any = None,
) -> Any:
    """Plot a reliability (calibration) diagram.

    Bins predictions into ``n_bins`` equal-width bins. For each bin, plots
    mean predicted confidence vs fraction of positives. A perfectly calibrated
    model lies on the diagonal.

    Args:
        y_true: Binary ground-truth labels.
        y_prob: Predicted probabilities.
        n_bins: Number of equal-width bins.
        ax: Optional matplotlib Axes. If None, a new Figure + Axes is created.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_means: list[float] = []
    bin_fracs: list[float] = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Last bin: include upper bound
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_means.append(float(y_prob[mask].mean()))
        bin_fracs.append(float(y_true[mask].mean()))

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfect calibration")
    ax.plot(bin_means, bin_fracs, marker="o", linewidth=1.5, color="steelblue", label="Model")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Reliability Diagram")
    ax.legend(fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    return fig


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    top_n: int = 30,
    ax: Any = None,
) -> Any:
    """Plot a horizontal bar chart of the top-N feature importances.

    Args:
        feature_names: Feature names aligned with ``importances``.
        importances: Importance scores (e.g. from ``model.feature_importances_``).
        top_n: Number of top features to display.
        ax: Optional matplotlib Axes. If None, a new Figure + Axes is created.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    importances = np.asarray(importances, dtype=np.float64)
    idx = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in idx]
    vals = importances[idx]

    fig = None
    if ax is None:
        height = max(4, top_n * 0.28)
        fig, ax = plt.subplots(figsize=(8, height))
    else:
        fig = ax.get_figure()

    ax.barh(range(len(names)), vals, color="steelblue", alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top-{top_n} Feature Importances")
    ax.invert_yaxis()  # most important at top

    return fig
