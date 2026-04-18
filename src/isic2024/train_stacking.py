"""Phase 3: Stacking meta-learner combining tabular and image OOF predictions.

Loads OOF predictions from Phase 1 (tabular GBDT ensemble) and Phase 2 (image DL
backbones), rank-transforms all features, and evaluates three stacking methods:
rank ensemble, logistic regression, and conservative LightGBM.

Usage::

    python -m isic2024.train_stacking --output-dir outputs/phase3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

from isic2024.evaluation.metrics import compute_pauc
from isic2024.models.ensemble import RankEnsemble

# Phase 2 backbone names and their output directories (relative to outputs/phase2/)
BACKBONES = ["efficientnetv2_s", "eva02", "convnextv2", "swin"]
RANK_COLS = ["rank_tabular", "rank_effnet", "rank_eva02", "rank_convnext", "rank_swin"]


def load_and_merge(
    phase1_path: Path,
    phase2_dir: Path,
    fold_path: Path,
) -> pd.DataFrame:
    """Load and merge all OOF predictions into a single DataFrame.

    Returns:
        DataFrame with columns: isic_id, target, fold, rank_tabular,
        rank_effnet, rank_eva02, rank_convnext, rank_swin.
    """
    # Phase 1 tabular predictions
    df_tab = pd.read_csv(phase1_path, usecols=["isic_id", "target", "ensemble"])
    logger.info(f"Phase 1 tabular: {len(df_tab)} rows")

    # Fold assignments
    df_folds = pd.read_csv(fold_path, usecols=["isic_id", "fold"])
    logger.info(f"Fold assignments: {len(df_folds)} rows")

    # Start with tabular + folds
    df = df_tab.merge(df_folds, on="isic_id", how="inner")

    # Phase 2 image predictions (one per backbone)
    pred_col_map = dict(zip(BACKBONES, ["pred_effnet", "pred_eva02", "pred_convnext", "pred_swin"]))
    for backbone in BACKBONES:
        oof_path = phase2_dir / backbone / "oof_image_predictions.csv"
        df_img = pd.read_csv(oof_path, usecols=["isic_id", "image_pred"])
        df_img = df_img.rename(columns={"image_pred": pred_col_map[backbone]})
        df = df.merge(df_img, on="isic_id", how="inner")
        logger.info(f"  {backbone}: {len(df_img)} rows, merged → {len(df)} rows")

    # Rank-transform all prediction columns → [0, 1]
    raw_cols = ["ensemble", "pred_effnet", "pred_eva02", "pred_convnext", "pred_swin"]
    for raw, rank in zip(raw_cols, RANK_COLS):
        df[rank] = rankdata(df[raw].values, method="average") / len(df)

    # Check for NaNs
    nan_counts = df[RANK_COLS + ["target", "fold"]].isna().sum()
    if nan_counts.any():
        logger.warning(f"NaN counts:\n{nan_counts[nan_counts > 0]}")

    logger.info(f"Final merged dataset: {len(df)} rows, {df['target'].sum():.0f} positives")
    return df


def run_rank_ensemble(df: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Simple rank-average of all 5 prediction sources."""
    predictions = [df[col].values for col in RANK_COLS]
    ens = RankEnsemble()
    oof = ens.predict(predictions)
    pauc = compute_pauc(df["target"].values, oof)
    return oof, pauc


def run_pauc_weighted_ensemble(
    df: pd.DataFrame, individual_paucs: dict
) -> tuple[np.ndarray, float]:
    """Rank-average weighted by each model's individual pAUC."""
    weights = [individual_paucs[col] for col in RANK_COLS]
    predictions = [df[col].values for col in RANK_COLS]
    ens = RankEnsemble(weights=weights)
    oof = ens.predict(predictions)
    pauc = compute_pauc(df["target"].values, oof)
    return oof, pauc


def run_logreg_stacker(df: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Logistic regression on rank features, 5-fold CV."""
    X = df[RANK_COLS].values
    y = df["target"].values
    folds = df["fold"].values
    oof = np.zeros(len(df), dtype=np.float64)

    for k in sorted(df["fold"].unique()):
        train_mask = folds != k
        val_mask = folds == k
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            C=1.0,
        )
        model.fit(X[train_mask], y[train_mask])
        oof[val_mask] = model.predict_proba(X[val_mask])[:, 1]

    pauc = compute_pauc(y, oof)
    return oof, pauc


def run_lgbm_stacker(
    df: pd.DataFrame,
    n_seeds: int = 3,
) -> tuple[np.ndarray, float]:
    """Conservative LightGBM stacker on rank features, 5-fold CV, seed-averaged."""
    X = df[RANK_COLS].values
    y = df["target"].values
    folds = df["fold"].values
    oof = np.zeros(len(df), dtype=np.float64)

    base_params = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 500,
        "learning_rate": 0.01,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 50,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "scale_pos_weight": 100,
        "verbose": -1,
        "n_jobs": -1,
    }

    for k in sorted(df["fold"].unique()):
        train_mask = folds != k
        val_mask = folds == k
        fold_preds = np.zeros(val_mask.sum(), dtype=np.float64)

        for seed in range(n_seeds):
            params = {**base_params, "random_state": seed}
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X[train_mask],
                y[train_mask],
                eval_set=[(X[val_mask], y[val_mask])],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            )
            fold_preds += model.predict_proba(X[val_mask])[:, 1]

        oof[val_mask] = fold_preds / n_seeds

    pauc = compute_pauc(y, oof)
    return oof, pauc


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Stacking meta-learner")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase3"),
        help="Directory for stacking outputs",
    )
    parser.add_argument(
        "--phase1-oof",
        type=Path,
        default=Path("outputs/oof_predictions.csv"),
        help="Phase 1 OOF predictions CSV",
    )
    parser.add_argument(
        "--phase2-dir",
        type=Path,
        default=Path("outputs/phase2"),
        help="Phase 2 output directory (contains backbone subdirs)",
    )
    parser.add_argument(
        "--fold-assignments",
        type=Path,
        default=Path("outputs/phase2/swin/fold_assignments.csv"),
        help="Fold assignments CSV",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load and merge
    df = load_and_merge(args.phase1_oof, args.phase2_dir, args.fold_assignments)

    # Compute individual pAUCs for reference and weighting
    individual_paucs = {}
    logger.info("Individual model pAUCs:")
    for col in RANK_COLS:
        p = compute_pauc(df["target"].values, df[col].values)
        individual_paucs[col] = p
        logger.info(f"  {col}: {p:.4f}")

    # Run all stacking methods
    results = {}

    # 1. Equal-weight rank ensemble
    oof_rank_ens, pauc_rank_ens = run_rank_ensemble(df)
    results["rank_ensemble"] = pauc_rank_ens
    logger.info(f"Rank ensemble (equal weights): pAUC = {pauc_rank_ens:.4f}")

    # 2. pAUC-weighted rank ensemble
    oof_weighted, pauc_weighted = run_pauc_weighted_ensemble(df, individual_paucs)
    results["rank_ensemble_weighted"] = pauc_weighted
    logger.info(f"Rank ensemble (pAUC-weighted): pAUC = {pauc_weighted:.4f}")

    # 3. Logistic regression
    oof_logreg, pauc_logreg = run_logreg_stacker(df)
    results["logreg_stacker"] = pauc_logreg
    logger.info(f"LogReg stacker: pAUC = {pauc_logreg:.4f}")

    # 4. LightGBM stacker
    oof_lgbm, pauc_lgbm = run_lgbm_stacker(df)
    results["lgbm_stacker"] = pauc_lgbm
    logger.info(f"LightGBM stacker: pAUC = {pauc_lgbm:.4f}")

    # Comparison table
    logger.info("\n" + "=" * 55)
    logger.info("STACKING COMPARISON")
    logger.info("=" * 55)
    logger.info(f"{'Method':<30} {'pAUC':>10}")
    logger.info("-" * 55)
    for col, p in individual_paucs.items():
        logger.info(f"  {col:<28} {p:>10.4f}")
    logger.info("-" * 55)
    for method, p in results.items():
        logger.info(f"  {method:<28} {p:>10.4f}")
    logger.info("=" * 55)

    best_method = max(results, key=results.get)
    logger.info(f"Best stacking method: {best_method} (pAUC = {results[best_method]:.4f})")

    # Save OOF predictions
    oof_df = pd.DataFrame({
        "isic_id": df["isic_id"],
        "target": df["target"],
        "rank_ensemble": oof_rank_ens,
        "rank_ensemble_weighted": oof_weighted,
        "logreg_stacker": oof_logreg,
        "lgbm_stacker": oof_lgbm,
    })
    oof_path = args.output_dir / "oof_stacking_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    logger.info(f"Saved OOF predictions to {oof_path}")

    # Save CV results
    cv_results = {
        "individual_paucs": {k: float(v) for k, v in individual_paucs.items()},
        "stacking_paucs": {k: float(v) for k, v in results.items()},
        "best_method": best_method,
        "best_pauc": float(results[best_method]),
        "n_samples": len(df),
        "n_positives": int(df["target"].sum()),
    }
    results_path = args.output_dir / "cv_results.json"
    results_path.write_text(json.dumps(cv_results, indent=2))
    logger.info(f"Saved CV results to {results_path}")


if __name__ == "__main__":
    main()
