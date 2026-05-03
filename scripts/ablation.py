"""Phase 3 Ablation Study — Leave-one-out, coefficient analysis, SHAP.

Loads all OOF predictions (1 tabular + 4 image), performs:
  1. Individual model pAUC baseline
  2. Leave-one-out ablation on the LogReg stacker (5-fold CV, same splits)
  3. LogReg coefficient analysis (fit on full OOF, no CV)
  4. SHAP attribution for the LogReg stacker (mean |SHAP| per source)
  5. Saves the final trained LogReg stacker (fit on all OOF) for use in web UI

Outputs (all under --output-dir):
    ablation_results.json        -- all numbers as JSON
    ablation_drop.png            -- pAUC-drop bar chart
    logreg_coefficients.png      -- coefficient bar chart
    shap_summary.png             -- mean |SHAP| per source
    logreg_stacker_final.pkl     -- trained LogReg for inference

Usage::

    python scripts/ablation.py
    python scripts/ablation.py --output-dir outputs/phase3
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import shap
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

from isic2024.evaluation.metrics import compute_pauc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BACKBONES = ["efficientnetv2_s", "eva02", "convnextv2", "swin"]
RANK_COLS = ["rank_tabular", "rank_effnet", "rank_eva02", "rank_convnext", "rank_swin"]

SOURCE_LABELS = {
    "rank_tabular":  "Tabular GBDT\n(Phase 1)",
    "rank_effnet":   "EfficientNetV2-S\n(20M, 224px)",
    "rank_eva02":    "EVA02-Small\n(22M, 336px)",
    "rank_convnext": "ConvNeXtV2-B\n(88M, 224px)",
    "rank_swin":     "SwinV2-B\n(87M, 256px)",
}

# Consistent colours: blue=tabular (ML), orange-family=image models (DL)
SOURCE_COLORS = {
    "rank_tabular":  "#2171b5",
    "rank_effnet":   "#fd8d3c",
    "rank_eva02":    "#e6550d",
    "rank_convnext": "#a1d99b",
    "rank_swin":     "#31a354",
}

LOGREG_KWARGS = dict(class_weight="balanced", max_iter=1000, solver="lbfgs", C=1.0)

FIGURE_DPI = 150
STYLE = "seaborn-v0_8-whitegrid"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_merged(
    phase1_path: Path,
    phase2_dir: Path,
    fold_path: Path,
) -> pd.DataFrame:
    """Merge Phase 1 tabular OOF + Phase 2 image OOFs on isic_id.

    Returns DataFrame with columns:
        isic_id, target, fold, rank_tabular, rank_effnet,
        rank_eva02, rank_convnext, rank_swin
    """
    df_tab = pd.read_csv(phase1_path, usecols=["isic_id", "target", "ensemble"])
    df_folds = pd.read_csv(fold_path, usecols=["isic_id", "fold"])
    df = df_tab.merge(df_folds, on="isic_id", how="inner")

    pred_col_map = dict(
        zip(BACKBONES, ["pred_effnet", "pred_eva02", "pred_convnext", "pred_swin"])
    )
    for backbone in BACKBONES:
        oof_path = phase2_dir / backbone / "oof_image_predictions.csv"
        df_img = pd.read_csv(oof_path, usecols=["isic_id", "image_pred"])
        df_img = df_img.rename(columns={"image_pred": pred_col_map[backbone]})
        df = df.merge(df_img, on="isic_id", how="inner")

    # Rank-transform all prediction columns → [0, 1]
    raw_cols = ["ensemble", "pred_effnet", "pred_eva02", "pred_convnext", "pred_swin"]
    for raw, rank in zip(raw_cols, RANK_COLS):
        df[rank] = rankdata(df[raw].values, method="average") / len(df)

    return df


# ---------------------------------------------------------------------------
# Ablation helpers
# ---------------------------------------------------------------------------

def _logreg_cv_pauc(df: pd.DataFrame, features: list[str]) -> float:
    """5-fold CV pAUC for a LogReg trained on `features`."""
    X = df[features].values
    y = df["target"].values
    folds = df["fold"].values
    oof = np.zeros(len(df), dtype=np.float64)

    for k in sorted(df["fold"].unique()):
        train_mask = folds != k
        val_mask = folds == k
        model = LogisticRegression(**LOGREG_KWARGS)
        model.fit(X[train_mask], y[train_mask])
        oof[val_mask] = model.predict_proba(X[val_mask])[:, 1]

    return compute_pauc(y, oof)


def run_individual_paucs(df: pd.DataFrame) -> dict[str, float]:
    return {col: compute_pauc(df["target"].values, df[col].values) for col in RANK_COLS}


def run_full_stacker_pauc(df: pd.DataFrame) -> float:
    return _logreg_cv_pauc(df, RANK_COLS)


def run_leave_one_out(df: pd.DataFrame, full_pauc: float) -> dict[str, dict]:
    """For each source, recompute LogReg pAUC with that source removed.

    Returns dict keyed by removed source with keys:
        pauc_without  -- CV pAUC using remaining 4 sources
        drop          -- full_pauc - pauc_without  (positive = that source helped)
        drop_pct      -- drop / full_pauc * 100
    """
    results = {}
    for col in RANK_COLS:
        remaining = [c for c in RANK_COLS if c != col]
        pauc_without = _logreg_cv_pauc(df, remaining)
        drop = full_pauc - pauc_without
        results[col] = {
            "pauc_without": round(pauc_without, 6),
            "drop": round(drop, 6),
            "drop_pct": round(drop / full_pauc * 100, 2),
        }
    return results


def fit_full_logreg(df: pd.DataFrame) -> LogisticRegression:
    """Fit LogReg on ALL OOF data (no CV) — for coefficient analysis and web UI."""
    X = df[RANK_COLS].values
    y = df["target"].values
    model = LogisticRegression(**LOGREG_KWARGS)
    model.fit(X, y)
    return model


def run_shap_analysis(df: pd.DataFrame, model: LogisticRegression) -> np.ndarray:
    """Compute SHAP values for the LogReg stacker using the shap library.

    For linear models, SHAP = LinearExplainer which gives exact attributions.

    Returns:
        shap_values: ndarray of shape (n_samples, n_features)
    """
    X = df[RANK_COLS].values
    explainer = shap.LinearExplainer(model, X, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X)
    # LinearExplainer returns values for the positive class directly
    return np.array(shap_values)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ablation_drop(loo_results: dict, full_pauc: float, out_path: Path) -> None:
    """Horizontal bar chart: pAUC drop when each source is removed."""
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.5))

        sources = list(loo_results.keys())
        drops = [loo_results[s]["drop"] for s in sources]
        labels = [SOURCE_LABELS[s].replace("\n", " ") for s in sources]
        colors = [SOURCE_COLORS[s] for s in sources]

        bars = ax.barh(labels, drops, color=colors, edgecolor="white", linewidth=0.5)

        # Annotate bars with exact values
        for bar, drop in zip(bars, drops):
            sign = "+" if drop >= 0 else ""
            ax.text(
                bar.get_width() + 0.0001,
                bar.get_y() + bar.get_height() / 2,
                f"{sign}{drop:.4f}",
                va="center",
                ha="left",
                fontsize=9,
            )

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("pAUC drop (full − without source)", fontsize=11)
        ax.set_title(
            f"Leave-One-Out Ablation  |  Full LogReg pAUC = {full_pauc:.4f}",
            fontsize=12,
            fontweight="bold",
        )
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
        ax.invert_yaxis()

        # Legend: ML vs DL
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#2171b5", label="ML (Tabular)"),
            Patch(facecolor="#fd8d3c", label="DL (Image)"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

        fig.tight_layout()
        fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


def plot_coefficients(model: LogisticRegression, out_path: Path) -> None:
    """Bar chart of LogReg stacker coefficients (fit on full OOF)."""
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.5))

        coefs = model.coef_[0]
        labels = [SOURCE_LABELS[c].replace("\n", " ") for c in RANK_COLS]
        colors = [SOURCE_COLORS[c] for c in RANK_COLS]

        bars = ax.barh(labels, coefs, color=colors, edgecolor="white", linewidth=0.5)
        for bar, coef in zip(bars, coefs):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{coef:.3f}",
                va="center",
                ha="left",
                fontsize=9,
            )

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("LogReg coefficient (higher = stronger positive contribution)", fontsize=10)
        ax.set_title(
            "LogReg Stacker Coefficients\n(fit on all OOF, class_weight=balanced)",
            fontsize=12,
            fontweight="bold",
        )
        ax.invert_yaxis()

        fig.tight_layout()
        fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


def plot_shap_summary(shap_values: np.ndarray, out_path: Path) -> None:
    """Bar chart of mean |SHAP| per source (global feature importance)."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.5))

        labels = [SOURCE_LABELS[c].replace("\n", " ") for c in RANK_COLS]
        colors = [SOURCE_COLORS[c] for c in RANK_COLS]

        # Sort by importance descending
        order = np.argsort(mean_abs_shap)
        sorted_labels = [labels[i] for i in order]
        sorted_vals = mean_abs_shap[order]
        sorted_colors = [colors[i] for i in order]

        bars = ax.barh(sorted_labels, sorted_vals, color=sorted_colors,
                       edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, sorted_vals):
            ax.text(
                bar.get_width() + 0.0001,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                ha="left",
                fontsize=9,
            )

        ax.set_xlabel("Mean |SHAP value| (average impact on log-odds)", fontsize=10)
        ax.set_title(
            "SHAP Feature Importance — LogReg Stacker\n"
            "(Which model source drives the malignancy prediction?)",
            fontsize=12,
            fontweight="bold",
        )

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#2171b5", label="ML (Tabular)"),
            Patch(facecolor="#fd8d3c", label="DL (Image)"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

        fig.tight_layout()
        fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(
    individual: dict[str, float],
    full_pauc: float,
    loo: dict[str, dict],
    coefs: np.ndarray,
) -> None:
    W = 65
    print("\n" + "=" * W)
    print("ABLATION STUDY — ISIC 2024 Phase 3 LogReg Stacker")
    print("=" * W)

    print("\n[1] Individual Model pAUC (no stacking):")
    print(f"  {'Source':<28} {'pAUC':>8}")
    print("  " + "-" * 36)
    for col, p in individual.items():
        label = SOURCE_LABELS[col].replace("\n", " ")
        marker = " <-- ML baseline" if col == "rank_tabular" else ""
        print(f"  {label:<28} {p:>8.4f}{marker}")

    print(f"\n[2] Full LogReg Stacker (5-fold CV): pAUC = {full_pauc:.4f}")
    print(f"    Gain over best individual: +{full_pauc - max(individual.values()):.4f}")

    print("\n[3] Leave-One-Out Ablation:")
    print(f"  {'Removed Source':<28} {'pAUC w/o':>8}  {'Drop':>8}  {'Drop %':>7}")
    print("  " + "-" * 56)
    for col, res in sorted(loo.items(), key=lambda x: -x[1]["drop"]):
        label = SOURCE_LABELS[col].replace("\n", " ")
        print(
            f"  {label:<28} {res['pauc_without']:>8.4f}  "
            f"{res['drop']:>+8.4f}  {res['drop_pct']:>6.1f}%"
        )

    print("\n[4] LogReg Stacker Coefficients (fit on full OOF):")
    print(f"  {'Source':<28} {'Coefficient':>12}")
    print("  " + "-" * 42)
    for col, coef in zip(RANK_COLS, coefs):
        label = SOURCE_LABELS[col].replace("\n", " ")
        print(f"  {label:<28} {coef:>12.4f}")

    print("=" * W + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 ablation study")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase3"))
    parser.add_argument("--phase1-oof", type=Path, default=Path("outputs/oof_predictions.csv"))
    parser.add_argument("--phase2-dir", type=Path, default=Path("outputs/phase2"))
    parser.add_argument(
        "--fold-assignments",
        type=Path,
        default=Path("outputs/phase2/swin/fold_assignments.csv"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and merging OOF predictions...")
    df = load_merged(args.phase1_oof, args.phase2_dir, args.fold_assignments)
    print(f"  {len(df):,} samples, {int(df['target'].sum())} positives\n")

    print("Computing individual pAUCs...")
    individual = run_individual_paucs(df)

    print("Computing full LogReg stacker pAUC (5-fold CV)...")
    full_pauc = run_full_stacker_pauc(df)

    print("Running leave-one-out ablation (5 × 5-fold CV)...")
    loo_results = run_leave_one_out(df, full_pauc)

    print("Fitting full LogReg on all OOF (for coefficients + SHAP + web UI)...")
    final_model = fit_full_logreg(df)

    print("Computing SHAP values (LinearExplainer)...")
    shap_values = run_shap_analysis(df, final_model)

    # Print diagnostic report
    print_report(individual, full_pauc, loo_results, final_model.coef_[0])

    # Save plots
    print("Saving figures...")
    plot_ablation_drop(loo_results, full_pauc, args.output_dir / "ablation_drop.png")
    plot_coefficients(final_model, args.output_dir / "logreg_coefficients.png")
    plot_shap_summary(shap_values, args.output_dir / "shap_summary.png")

    # Save trained LogReg for web UI
    model_path = args.output_dir / "logreg_stacker_final.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": final_model, "feature_cols": RANK_COLS}, f)
    print(f"  Saved: {model_path}")

    # Save all numbers as JSON
    results = {
        "individual_paucs": {k: round(v, 6) for k, v in individual.items()},
        "full_logreg_pauc": round(full_pauc, 6),
        "gain_over_best_individual": round(full_pauc - max(individual.values()), 6),
        "leave_one_out": loo_results,
        "logreg_coefficients": {
            col: round(float(coef), 6)
            for col, coef in zip(RANK_COLS, final_model.coef_[0])
        },
        "shap_mean_abs": {
            col: round(float(v), 6)
            for col, v in zip(RANK_COLS, np.abs(shap_values).mean(axis=0))
        },
        "n_samples": len(df),
        "n_positives": int(df["target"].sum()),
    }
    json_path = args.output_dir / "ablation_results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"  Saved: {json_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
