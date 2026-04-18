"""Generate report figures from OOF prediction CSVs.

Produces three PDF figures:
  1. phase2_roc_curves.pdf   — ROC curves for 4 image backbones with pAUC shaded region
  2. phase2_pauc_bar.pdf     — pAUC bar chart across all phases
  3. stacking_comparison.pdf — Phase 3 stacking horizontal bar chart

Usage:
    python -m report.generate_figures
    # or
    python report/generate_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

# Add project root to path so we can import isic2024
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isic2024.evaluation.metrics import compute_pauc  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = PROJECT_ROOT / "report" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

PHASE1_OOF = OUTPUTS / "oof_predictions.csv"
PHASE2_DIRS = {
    "EfficientNetV2-S": OUTPUTS / "phase2" / "efficientnetv2_s",
    "EVA02": OUTPUTS / "phase2" / "eva02",
    "ConvNeXtV2-B": OUTPUTS / "phase2" / "convnextv2",
    "SwinV2-B": OUTPUTS / "phase2" / "swin",
}
PHASE3_RESULTS = OUTPUTS / "phase3" / "cv_results.json"

# ---------------------------------------------------------------------------
# Style — clean, IEEE column-width (~3.5 in), colorblind-friendly
# ---------------------------------------------------------------------------
# Wong (2011) colorblind-safe palette
CB_BLUE = "#0072B2"
CB_ORANGE = "#E69F00"
CB_GREEN = "#009E73"
CB_VERMILLION = "#D55E00"
CB_SKY = "#56B4E9"
CB_PINK = "#CC79A7"
CB_YELLOW = "#F0E442"
CB_BLACK = "#000000"

PHASE_COLORS = {
    "Phase 1": CB_BLUE,
    "Phase 2": CB_ORANGE,
    "Phase 3": CB_GREEN,
}

BACKBONE_COLORS = [CB_BLUE, CB_ORANGE, CB_GREEN, CB_VERMILLION]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ---------------------------------------------------------------------------
# Helper: load Phase 2 OOF predictions
# ---------------------------------------------------------------------------
def load_phase2_oof() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {backbone_name: (y_true, y_pred)} from OOF CSVs."""
    data = {}
    for name, d in PHASE2_DIRS.items():
        csv_path = d / "oof_image_predictions.csv"
        df = pd.read_csv(csv_path)
        data[name] = (df["target"].values, df["image_pred"].values)
    return data


# ===================================================================
# Figure 1: Phase 2 ROC curves with pAUC shaded region
# ===================================================================
def fig_phase2_roc():
    phase2 = load_phase2_oof()

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    # The pAUC region corresponds to TPR >= 0.80, i.e. FPR <= 0.20 in
    # the *original* (non-flipped) ROC space.  We shade FPR in [0, 0.20].
    fpr_shade = np.linspace(0, 0.20, 200)

    for (name, (y_true, y_pred)), color in zip(phase2.items(), BACKBONE_COLORS):
        pauc = compute_pauc(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        ax.plot(fpr, tpr, color=color, linewidth=1.2, label=f"{name} (pAUC={pauc:.4f})")

    # Shade the pAUC region (FPR 0-0.20) using the diagonal as lower bound
    ax.axvspan(0, 0.20, alpha=0.08, color="grey", zorder=0)
    ax.axvline(0.20, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.text(0.10, 0.05, "pAUC\nregion", ha="center", va="bottom", fontsize=6, color="grey")

    # Diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, alpha=0.3)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Phase 2: Image Backbone ROC Curves (OOF)")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)

    out = FIGURES / "phase2_roc_curves.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ===================================================================
# Figure 2: pAUC bar chart across all phases
# ===================================================================
def fig_pauc_bar():
    # Phase 1 individual models
    p1 = pd.read_csv(PHASE1_OOF)
    y_true_p1 = p1["target"].values
    phase1_models = {
        "LightGBM": compute_pauc(y_true_p1, p1["lgbm"].values),
        "XGBoost": compute_pauc(y_true_p1, p1["xgb"].values),
        "CatBoost": compute_pauc(y_true_p1, p1["catboost"].values),
        "GBDT Ensemble": compute_pauc(y_true_p1, p1["ensemble_calibrated"].values),
    }

    # Phase 2
    phase2 = load_phase2_oof()
    phase2_models = {name: compute_pauc(yt, yp) for name, (yt, yp) in phase2.items()}

    # Phase 3
    with open(PHASE3_RESULTS) as f:
        p3 = json.load(f)
    phase3_models = {
        "Rank Ensemble": p3["stacking_paucs"]["rank_ensemble"],
        "LogReg Stacker": p3["stacking_paucs"]["logreg_stacker"],
    }

    # Build ordered lists
    labels, values, colors = [], [], []
    for name, v in phase1_models.items():
        labels.append(name)
        values.append(v)
        colors.append(PHASE_COLORS["Phase 1"])
    for name, v in phase2_models.items():
        labels.append(name)
        values.append(v)
        colors.append(PHASE_COLORS["Phase 2"])
    for name, v in phase3_models.items():
        labels.append(name)
        values.append(v)
        colors.append(PHASE_COLORS["Phase 3"])

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.65, edgecolor="white", linewidth=0.3)

    # Value labels on bars
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=5.5,
            rotation=45,
        )

    # Target line
    ax.axhline(0.16, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(len(labels) - 0.5, 0.1605, "target = 0.16", ha="right", fontsize=6, color="red")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("pAUC (TPR $\\geq$ 0.80)")
    ax.set_title("pAUC Across All Models and Phases")
    ax.set_ylim(0, max(values) * 1.15)

    # Phase legend (manual patches)
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=PHASE_COLORS["Phase 1"], label="Phase 1 (Tabular)"),
        Patch(facecolor=PHASE_COLORS["Phase 2"], label="Phase 2 (Image)"),
        Patch(facecolor=PHASE_COLORS["Phase 3"], label="Phase 3 (Stacking)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=6)

    out = FIGURES / "phase2_pauc_bar.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ===================================================================
# Figure 3: Stacking comparison horizontal bar chart
# ===================================================================
def fig_stacking_comparison():
    with open(PHASE3_RESULTS) as f:
        p3 = json.load(f)

    # Individual rank sources
    indiv = p3["individual_paucs"]
    # Stacking methods
    stack = p3["stacking_paucs"]

    # Pretty names
    indiv_names = {
        "rank_tabular": "Tabular (GBDT)",
        "rank_effnet": "EfficientNetV2-S",
        "rank_eva02": "EVA02",
        "rank_convnext": "ConvNeXtV2-B",
        "rank_swin": "SwinV2-B",
    }
    stack_names = {
        "rank_ensemble": "Rank Ensemble",
        "rank_ensemble_weighted": "Rank Ens. (weighted)",
        "logreg_stacker": "LogReg Stacker",
        "lgbm_stacker": "LGBM Stacker",
    }

    labels, values, group = [], [], []
    for k, pretty in indiv_names.items():
        labels.append(pretty)
        values.append(indiv[k])
        group.append("individual")
    for k, pretty in stack_names.items():
        labels.append(pretty)
        values.append(stack[k])
        group.append("stacking")

    best_idx = int(np.argmax(values))

    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    y = np.arange(len(labels))

    bar_colors = []
    for i, g in enumerate(group):
        if i == best_idx:
            bar_colors.append(CB_VERMILLION)
        elif g == "individual":
            bar_colors.append(CB_SKY)
        else:
            bar_colors.append(CB_GREEN)

    bars = ax.barh(y, values, color=bar_colors, height=0.6, edgecolor="white", linewidth=0.3)

    # Value labels
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_width() + 0.0008,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.4f}",
            ha="left",
            va="center",
            fontsize=6,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("pAUC (TPR $\\geq$ 0.80)")
    ax.set_title("Phase 3: Stacking Comparison")
    ax.invert_yaxis()

    # Divider between individual and stacking
    n_indiv = len(indiv_names)
    ax.axhline(n_indiv - 0.5, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.text(
        min(values) * 0.98,
        n_indiv - 0.7,
        "Individual",
        fontsize=6,
        color="grey",
        va="bottom",
    )
    ax.text(
        min(values) * 0.98,
        n_indiv - 0.3,
        "Stacking",
        fontsize=6,
        color="grey",
        va="top",
    )

    # Highlight best
    ax.text(
        values[best_idx] + 0.0008,
        best_idx + 0.25,
        "best",
        fontsize=5.5,
        color=CB_VERMILLION,
        fontweight="bold",
        va="top",
    )

    ax.set_xlim(min(values) * 0.95, max(values) * 1.06)

    out = FIGURES / "stacking_comparison.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print("Generating report figures...")
    fig_phase2_roc()
    fig_pauc_bar()
    fig_stacking_comparison()
    print("Done.")
