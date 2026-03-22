"""End-to-end training pipeline for ISIC 2024.

Orchestrates:
    1. Data loading
    2. StratifiedGroupKFold split creation (no patient leakage)
    3. Per-fold feature pipeline (preprocess → engineer → ugly duckling → select)
    4. Per-fold training for LightGBM, XGBoost, CatBoost (with seed averaging), SVM
    5. Rank ensemble of all OOF predictions
    6. Isotonic calibration on ensemble OOF
    7. OOF pAUC + full metrics logging
    8. Saving OOF predictions, CV results, and inference artifacts
       (full-data preprocessor, selector, feature_names)

Usage::

    python -m isic2024.train                           # uses configs/base.yaml
    python -m isic2024.train --config path/to/cfg.yaml
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedGroupKFold

from isic2024.config import Config
from isic2024.data.loader import load_data
from isic2024.evaluation.metrics import compute_metrics
from isic2024.features.pipeline import build_feature_pipeline
from isic2024.models.calibration import calibrator_factory
from isic2024.models.ensemble import RankEnsemble
from isic2024.models.gbdt import model_factory
from isic2024.models.svm_baseline import SVMBaseline


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Orchestrates the full CV training loop.

    Args:
        config: Project configuration dataclass.
        output_dir: Directory for saving OOF predictions, CV results, and
            inference artifacts. Defaults to ``outputs/``.
    """

    def __init__(self, config: Config, output_dir: str | Path = "outputs") -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run the full pipeline end-to-end.

        Steps:
            1. Create folds from raw df (no pipeline needed — just target + patient_id).
            2. CV training: per-fold pipelines + model fits + OOF collection.
            3. Full-data pipeline on all rows → inference artifacts.
            4. Save all outputs.

        Args:
            df: Raw training DataFrame (un-preprocessed).

        Returns:
            Dict with ``oof_df``, ``cv_results``, ``feature_names``.
        """
        cfg = self.config

        # Step 1 — folds from raw df (no pipeline required here)
        folds = self.create_folds(df)

        # Step 2 — CV training (each fold independently fits its own pipeline)
        results = self.train_cv(df, folds)

        # Step 3 — Full-data pipeline for inference artifacts.
        # This is the pipeline that would be used to preprocess the test set,
        # so we fit it on ALL training data (not per-fold).
        logger.info("Fitting full-data pipeline for inference artifacts...")
        _, feature_names, preprocessor, selector = build_feature_pipeline(
            df.copy(), cfg, is_train=True
        )
        results["feature_names"] = feature_names
        logger.info(f"Full-data feature count: {len(feature_names)}")

        # Step 4 — Save
        self._save_results(results, feature_names, preprocessor, selector)

        return results

    def create_folds(
        self,
        df: pd.DataFrame,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Create StratifiedGroupKFold splits; validate patient separation.

        Works directly on the raw DataFrame — only requires ``target_col`` and
        ``patient_col`` columns (no preprocessing needed).

        Args:
            df: DataFrame containing at minimum ``target_col`` and
                ``patient_col`` columns.

        Returns:
            List of ``(train_idx, val_idx)`` positional index arrays (one per fold).

        Raises:
            ValueError: If any patient appears in both train and val of a fold,
                or if any val fold contains zero positive examples.
        """
        cfg = self.config
        target_col = cfg.data.target_col
        patient_col = cfg.data.patient_col

        y = df[target_col].values
        groups = df[patient_col].values

        sgkf = StratifiedGroupKFold(
            n_splits=cfg.cv.n_splits,
            shuffle=True,
            random_state=cfg.cv.seed,
        )

        folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(
            sgkf.split(np.zeros(len(df)), y, groups)
        ):
            # Guard: no patient should appear in both splits
            train_patients = set(groups[train_idx])
            val_patients = set(groups[val_idx])
            overlap = train_patients & val_patients
            if overlap:
                raise ValueError(
                    f"Fold {fold_idx}: patient overlap detected: {overlap}"
                )

            n_pos_train = int(y[train_idx].sum())
            n_pos_val = int(y[val_idx].sum())
            logger.info(
                f"Fold {fold_idx}: train={len(train_idx):,} "
                f"(pos={n_pos_train}) | val={len(val_idx):,} (pos={n_pos_val})"
            )

            if n_pos_val == 0:
                raise ValueError(
                    f"Fold {fold_idx} has 0 positive examples in validation. "
                    "Cannot compute pAUC."
                )

            folds.append((train_idx, val_idx))

        return folds

    def train_cv(
        self,
        df_raw: pd.DataFrame,
        folds: list[tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, Any]:
        """Run cross-validation for all models with seed averaging.

        For each fold:
          - Re-fits the full feature pipeline on the fold's training split only
            (prevents leakage of validation statistics into feature engineering
            and feature selection).
          - Trains LightGBM, XGBoost, CatBoost with seed averaging and SVM.
          - Collects OOF predictions.

        After all folds:
          - Builds a RankEnsemble over all OOF predictions.
          - Fits an IsotonicCalibrator on ensemble OOF.
          - Logs per-model and ensemble pAUC / AUC.

        Args:
            df_raw: Original un-preprocessed DataFrame.
            folds: List of ``(train_idx, val_idx)`` from ``create_folds``.

        Returns:
            Dict with:
                - ``oof_df``: DataFrame with ``target`` + per-model OOF columns.
                - ``cv_results``: Per-fold and mean metrics per model.
        """
        cfg = self.config
        target_col = cfg.data.target_col
        n = len(df_raw)

        # Target values are never changed by preprocessing — grab them now.
        y_true = df_raw[target_col].values.astype(np.float64)

        model_names = ["lgbm", "xgb", "catboost", "svm"]
        oof_preds: dict[str, np.ndarray] = {m: np.zeros(n) for m in model_names}
        fold_metrics: dict[str, list[dict[str, float]]] = {m: [] for m in model_names}

        seeds = cfg.seed_averaging.seeds if cfg.seed_averaging.enabled else [cfg.seed]
        gbdt_names = ["lgbm", "xgb", "catboost"]

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            logger.info("=" * 60)
            logger.info(f"Fold {fold_idx + 1}/{cfg.cv.n_splits}")

            # Re-fit the full pipeline on this fold's training split only.
            df_fold_train_raw = df_raw.iloc[train_idx].copy()
            df_fold_val_raw = df_raw.iloc[val_idx].copy()

            df_tr, fold_features, fold_preprocessor, fold_selector = (
                build_feature_pipeline(df_fold_train_raw, cfg, is_train=True)
            )
            df_va, _, _, _ = build_feature_pipeline(
                df_fold_val_raw,
                cfg,
                preprocessor=fold_preprocessor,
                selector=fold_selector,
                is_train=False,
            )

            X_tr = df_tr[fold_features].values.astype(np.float32)
            y_tr = df_tr[target_col].values.astype(np.float64)
            X_va = df_va[fold_features].values.astype(np.float32)
            y_va = df_va[target_col].values.astype(np.float64)

            # ---- GBDT models with seed averaging ----
            for model_name in gbdt_names:
                model_cfg = getattr(cfg, model_name)
                seed_preds = []
                for seed in seeds:
                    model = model_factory(model_name, model_cfg)
                    model.fit(X_tr, y_tr, X_va, y_va, seed=seed)
                    seed_preds.append(model.predict_proba(X_va))

                fold_pred = np.mean(seed_preds, axis=0)
                oof_preds[model_name][val_idx] = fold_pred
                metrics = compute_metrics(y_va, fold_pred)
                fold_metrics[model_name].append(metrics)
                logger.info(
                    f"  {model_name.upper():8s} fold {fold_idx}: "
                    f"pAUC={metrics['pauc']:.4f}  AUC={metrics['roc_auc']:.4f}"
                )

            # ---- SVM (single seed — subsamples internally for speed) ----
            svm_model = SVMBaseline(cfg.svm)
            svm_model.fit(X_tr, y_tr, X_va, y_va, seed=cfg.seed)
            svm_pred = svm_model.predict_proba(X_va)
            oof_preds["svm"][val_idx] = svm_pred
            svm_metrics = compute_metrics(y_va, svm_pred)
            fold_metrics["svm"].append(svm_metrics)
            logger.info(
                f"  {'SVM':8s} fold {fold_idx}: "
                f"pAUC={svm_metrics['pauc']:.4f}  AUC={svm_metrics['roc_auc']:.4f}"
            )

        # ---- Ensemble OOF (GBDT only — SVM excluded: high variance hurts blend) ----
        gbdt_names_ensemble = ["lgbm", "xgb", "catboost"]
        ensemble = RankEnsemble()
        ensemble_pred = ensemble.predict([oof_preds[m] for m in gbdt_names_ensemble])
        ensemble_metrics = compute_metrics(y_true, ensemble_pred)

        # ---- Calibration on full OOF (valid: each prediction was out-of-fold) ----
        calibrator = calibrator_factory(cfg.calibration.method)
        calibrator.fit(y_true, ensemble_pred)
        calibrated_pred = calibrator.transform(ensemble_pred)
        calibrated_metrics = compute_metrics(y_true, calibrated_pred)

        # ---- Summary table ----
        logger.info("\n" + "=" * 60)
        logger.info("CV RESULTS SUMMARY")
        logger.info("=" * 60)

        cv_results: dict[str, Any] = {}
        for model_name in model_names:
            paucs = [m["pauc"] for m in fold_metrics[model_name]]
            aucs = [m["roc_auc"] for m in fold_metrics[model_name]]
            cv_results[model_name] = {
                "fold_metrics": fold_metrics[model_name],
                "mean_pauc": float(np.mean(paucs)),
                "std_pauc": float(np.std(paucs)),
                "mean_roc_auc": float(np.mean(aucs)),
                "std_roc_auc": float(np.std(aucs)),
            }
            logger.info(
                f"  {model_name.upper():8s}  "
                f"pAUC={np.mean(paucs):.4f}±{np.std(paucs):.4f}  "
                f"AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}"
            )

        cv_results["ensemble"] = {"oof_metrics": ensemble_metrics}
        cv_results["ensemble_calibrated"] = {"oof_metrics": calibrated_metrics}

        logger.info(
            f"  {'ENSEMBLE':8s}  pAUC={ensemble_metrics['pauc']:.4f}  "
            f"AUC={ensemble_metrics['roc_auc']:.4f}"
        )
        logger.info(
            f"  {'CALIB':8s}  pAUC={calibrated_metrics['pauc']:.4f}  "
            f"ECE={calibrated_metrics['ece']:.4f}"
        )

        # ---- OOF DataFrame — include isic_id for auditability if present ----
        oof_data: dict[str, Any] = {
            target_col: y_true,
            **{m: oof_preds[m] for m in model_names},
            "ensemble": ensemble_pred,
            "ensemble_calibrated": calibrated_pred,
        }
        oof_df = pd.DataFrame(oof_data)
        if "isic_id" in df_raw.columns:
            oof_df.insert(0, "isic_id", df_raw["isic_id"].values)

        return {
            "oof_df": oof_df,
            "cv_results": cv_results,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_results(
        self,
        results: dict[str, Any],
        feature_names: list[str],
        preprocessor: Any,
        selector: Any,
    ) -> None:
        """Persist all outputs needed for evaluation and inference."""
        # OOF predictions
        oof_path = self.output_dir / "oof_predictions.csv"
        results["oof_df"].to_csv(oof_path, index=False)
        logger.info(f"OOF predictions saved → {oof_path}")

        # CV metrics
        cv_serialisable = _to_serialisable(results["cv_results"])
        cv_path = self.output_dir / "cv_results.json"
        cv_path.write_text(json.dumps(cv_serialisable, indent=2))
        logger.info(f"CV results saved → {cv_path}")

        # Feature names (from full-data pipeline — use for test-set inference)
        feat_path = self.output_dir / "feature_names.json"
        feat_path.write_text(json.dumps(feature_names, indent=2))
        logger.info(f"Feature names saved → {feat_path}")

        # Inference artifacts: preprocessor + selector (fit on full training data)
        prep_path = self.output_dir / "preprocessor.pkl"
        with open(prep_path, "wb") as f:
            pickle.dump(preprocessor, f)
        logger.info(f"Preprocessor saved → {prep_path}")

        sel_path = self.output_dir / "selector.pkl"
        with open(sel_path, "wb") as f:
            pickle.dump(selector, f)
        logger.info(f"Selector saved → {sel_path}")


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

def _to_serialisable(obj: Any) -> Any:
    """Recursively convert numpy scalars / arrays to Python native types."""
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serialisable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI: python -m isic2024.train [--config ...] [--output-dir ...]"""
    parser = argparse.ArgumentParser(description="ISIC 2024 training pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parents[2] / "configs" / "base.yaml"),
        help="Path to YAML config (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results and artifacts (default: outputs/)",
    )
    args = parser.parse_args()

    logger.info(f"Loading config from {args.config}")
    config = Config.from_yaml(args.config)

    logger.info(f"Loading data from {config.data.train_path}")
    df = load_data(config.data.train_path)

    trainer = Trainer(config, output_dir=args.output_dir)
    results = trainer.run(df)

    ensemble_pauc = results["cv_results"]["ensemble"]["oof_metrics"]["pauc"]
    logger.info(f"\nFinal ensemble OOF pAUC: {ensemble_pauc:.4f}")


if __name__ == "__main__":
    main()
