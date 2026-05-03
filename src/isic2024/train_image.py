"""Phase 2 training: image-based deep learning with Lightning.

Runs StratifiedGroupKFold CV (identical splits to Phase 1), trains a timm
backbone per fold, performs TTA prediction, and saves OOF predictions for
Tier 3 stacking.

Usage::

    python -m isic2024.train_image --config configs/phase2.yaml --folds 0
    python -m isic2024.train_image --config configs/phase2.yaml --folds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import gc
import pickle
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from isic2024.config_phase2 import Phase2Config
from isic2024.data.augmentation import get_tta_transforms
from isic2024.data.image_dataset import ISICDataModule, ISICImageDataset, _worker_init_fn
from isic2024.evaluation.metrics import compute_metrics
from isic2024.models.image_module import ISICImageModule

# Fixed categorical mappings — must be consistent across train/val/test splits.
# Derived from the full ISIC 2024 training set; -1 = missing/unknown.
_CAT_MAPS: dict[str, dict[str, int]] = {
    "sex": {"male": 0, "female": 1},
    "anatom_site_general": {
        "anterior torso": 0, "head/neck": 1, "lower extremity": 2,
        "posterior torso": 3, "upper extremity": 4,
    },
    "tbp_tile_type": {"3D: white": 0, "3D: XP": 1},
}


def prepare_tabular(
    df: pd.DataFrame,
    features: list[str],
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, StandardScaler]:
    """Extract, impute, and scale tabular features for one data split.

    Handles:
      - String categoricals (sex, anatom_site_general, tbp_tile_type) via integer encoding
      - n_lesions_patient: computed inline from patient_id if not present
      - Missing columns: zero-filled with a warning

    Args:
        df: DataFrame containing the raw metadata columns.
        features: Ordered list of column names to use.
        scaler: Pre-fit StandardScaler (for val/test). If None, fits a new one.

    Returns:
        (matrix, scaler) where matrix is float32 (N, len(features)).
    """
    df = df.copy()

    # Compute n_lesions_patient inline if needed (ugly-duckling patient context)
    if "n_lesions_patient" in features and "n_lesions_patient" not in df.columns:
        if "patient_id" in df.columns:
            counts = df.groupby("patient_id")["patient_id"].transform("count")
            df["n_lesions_patient"] = counts.astype(np.float32)
        else:
            df["n_lesions_patient"] = 1.0

    # Encode string categoricals with fixed mappings (consistent across splits).
    # Using dynamic pd.Categorical codes would assign different integers if a
    # category is absent from a split, breaking the StandardScaler alignment.
    for col, mapping in _CAT_MAPS.items():
        if col in df.columns and pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].map(mapping).fillna(-1).astype(np.float32)

    # Build feature matrix (zero-fill any missing columns)
    missing = [f for f in features if f not in df.columns]
    if missing:
        logger.warning(
            f"Tabular conditioning: {len(missing)} features not found, zero-filled: {missing}"
        )

    matrix = np.zeros((len(df), len(features)), dtype=np.float32)
    for i, feat in enumerate(features):
        if feat in df.columns:
            col_vals = pd.to_numeric(df[feat], errors="coerce").values.astype(np.float32)
            col_vals = np.nan_to_num(col_vals, nan=0.0)
            matrix[:, i] = col_vals

    if scaler is None:
        scaler = StandardScaler()
        matrix = scaler.fit_transform(matrix).astype(np.float32)
    else:
        matrix = scaler.transform(matrix).astype(np.float32)

    return matrix, scaler


def create_folds(
    df: pd.DataFrame,
    cfg: Phase2Config,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create StratifiedGroupKFold splits identical to Phase 1."""
    y = df[cfg.data.target_col].values
    groups = df[cfg.data.patient_col].values

    sgkf = StratifiedGroupKFold(
        n_splits=cfg.cv.n_splits,
        shuffle=True,
        random_state=cfg.cv.seed,
    )

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        sgkf.split(np.zeros(len(df)), y, groups)
    ):
        train_patients = set(groups[train_idx])
        val_patients = set(groups[val_idx])
        overlap = train_patients & val_patients
        if overlap:
            raise ValueError(f"Fold {fold_idx}: patient overlap: {overlap}")

        n_pos_val = int(y[val_idx].sum())
        if n_pos_val == 0:
            raise ValueError(f"Fold {fold_idx}: 0 positives in validation")

        logger.info(
            f"Fold {fold_idx}: train={len(train_idx):,} "
            f"(pos={int(y[train_idx].sum())}) | "
            f"val={len(val_idx):,} (pos={n_pos_val})"
        )
        folds.append((train_idx, val_idx))

    return folds


def predict_tta(
    model: ISICImageModule,
    val_df: pd.DataFrame,
    cfg: Phase2Config,
    trainer: L.Trainer,
    val_tabular: np.ndarray | None = None,
) -> np.ndarray:
    """Run TTA with 8 D4 transforms, average sigmoid probabilities."""
    tta_transforms = get_tta_transforms(
        cfg.augment, cfg.image.size,
        cfg.image.normalize_mean, cfg.image.normalize_std,
    )

    all_probs = []
    for tfm in tta_transforms:
        ds = ISICImageDataset(
            val_df, cfg.image.hdf5_path, tfm, cfg.data.target_col,
            image_size=cfg.image.size, tabular_matrix=val_tabular,
        )
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=cfg.train.batch_size * 2,
            shuffle=False,
            num_workers=cfg.train.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=_worker_init_fn,
        )
        preds = trainer.predict(model, dl)
        logits = torch.cat(preds)
        all_probs.append(logits.sigmoid().numpy())

    return np.mean(all_probs, axis=0)


def train_fold(
    fold_idx: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    df: pd.DataFrame,
    cfg: Phase2Config,
    output_dir: Path,
    resume: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Train a single fold, return (val_indices, oof_predictions)."""
    L.seed_everything(cfg.seed + fold_idx)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # Tabular conditioning: preprocess features per fold (no leakage)
    train_tabular, val_tabular = None, None
    if cfg.tabular.enabled:
        train_tabular, fold_scaler = prepare_tabular(train_df, cfg.tabular.features)
        val_tabular, _ = prepare_tabular(val_df, cfg.tabular.features, scaler=fold_scaler)
        logger.info(
            f"Tabular conditioning: {len(cfg.tabular.features)} features, "
            f"train shape {train_tabular.shape}"
        )

    # DataModule
    dm = ISICDataModule(cfg, train_df, val_df, train_tabular, val_tabular)

    # Model
    model = ISICImageModule(cfg)

    # Callbacks
    ckpt_dir = output_dir / f"fold_{fold_idx}" / "checkpoints"
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor="val/pauc",
        mode="max",
        save_top_k=cfg.train.save_top_k,
        save_last=True,
        filename="epoch={epoch}-pauc={val/pauc:.4f}",
        auto_insert_metric_name=False,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/pauc",
        mode="max",
        patience=cfg.train.patience,
    )

    # Logger
    wandb_logger = None
    if cfg.wandb.enabled:
        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"fold_{fold_idx}_{cfg.model.backbone}",
            group=cfg.model.backbone,
        )

    # Trainer
    trainer = L.Trainer(
        max_epochs=cfg.train.epochs,
        precision=cfg.train.precision,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=wandb_logger or True,
        fast_dev_run=cfg.train.fast_dev_run,
        enable_progress_bar=True,
        deterministic=False,
    )

    # Train
    ckpt_path = "last" if resume and (ckpt_dir / "last.ckpt").exists() else None
    trainer.fit(model, dm, ckpt_path=ckpt_path)

    # Load best checkpoint for prediction
    best_path = checkpoint_cb.best_model_path
    if best_path:
        logger.info(f"Loading best checkpoint: {best_path}")
        model = ISICImageModule.load_from_checkpoint(best_path, cfg=cfg)

    # TTA prediction (skip full TTA during fast_dev_run — just return zeros)
    model.eval()
    if cfg.train.fast_dev_run:
        logger.info("fast_dev_run: skipping TTA, returning dummy predictions")
        oof_preds = np.zeros(len(val_df), dtype=np.float64)
    else:
        # Create a separate trainer for prediction without fast_dev_run limit
        predict_trainer = L.Trainer(
            precision=cfg.train.precision,
            enable_progress_bar=True,
            logger=False,
        )
        oof_preds = predict_tta(model, val_df, cfg, predict_trainer, val_tabular)

        y_val = val_df[cfg.data.target_col].values
        metrics = compute_metrics(y_val, oof_preds)
        logger.info(
            f"Fold {fold_idx} TTA: pAUC={metrics['pauc']:.4f} "
            f"AUC={metrics['roc_auc']:.4f} Brier={metrics['brier']:.4f}"
        )

    # Cleanup
    if wandb_logger:
        import wandb
        wandb.finish()
    del model, trainer, dm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return val_idx, oof_preds


def main() -> None:
    parser = argparse.ArgumentParser(description="ISIC 2024 Phase 2: Image Training")
    parser.add_argument(
        "--config", type=str,
        default=str(Path(__file__).parents[2] / "configs" / "phase2.yaml"),
    )
    parser.add_argument("--output-dir", type=str, default="outputs/phase2")
    parser.add_argument("--folds", type=str, default="0,1,2,3,4",
                        help="Comma-separated fold indices, e.g. '0' or '0,1,2,3,4'")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint if available")
    args = parser.parse_args()

    cfg = Phase2Config.from_yaml(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_indices = [int(f) for f in args.folds.split(",")]

    logger.info(f"Config: {args.config}")
    logger.info(f"Backbone: {cfg.model.backbone}, Image size: {cfg.image.size}")
    logger.info(f"Folds to train: {fold_indices}")

    # Load metadata
    df = pd.read_csv(cfg.data.train_path)
    logger.info(f"Loaded {len(df):,} samples, {df[cfg.data.target_col].sum():.0f} positives")

    # Save full-data tabular scaler for inference (fit on all training rows)
    if cfg.tabular.enabled:
        _, full_scaler = prepare_tabular(df, cfg.tabular.features)
        scaler_path = output_dir / "tabular_scaler.pkl"
        with open(scaler_path, "wb") as fh:
            pickle.dump({"scaler": full_scaler, "features": cfg.tabular.features}, fh)
        logger.info(f"Full-data tabular scaler saved → {scaler_path}")

    # Create folds
    folds = create_folds(df, cfg)

    # Save fold assignments
    fold_assign = pd.DataFrame({"isic_id": df["isic_id"]})
    fold_assign["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(folds):
        fold_assign.loc[val_idx, "fold"] = fold_idx
    fold_assign.to_csv(output_dir / "fold_assignments.csv", index=False)
    logger.info(f"Fold assignments saved → {output_dir / 'fold_assignments.csv'}")

    # Train selected folds
    n = len(df)
    oof_preds = np.full(n, np.nan, dtype=np.float64)
    oof_mask = np.zeros(n, dtype=bool)

    for fold_idx in fold_indices:
        logger.info("=" * 60)
        logger.info(f"Training fold {fold_idx}/{cfg.cv.n_splits - 1}")
        logger.info("=" * 60)

        train_idx, val_idx = folds[fold_idx]
        vidx, preds = train_fold(
            fold_idx, train_idx, val_idx, df, cfg, output_dir, args.resume,
        )
        oof_preds[vidx] = preds
        oof_mask[vidx] = True

    # Save OOF predictions
    oof_df = pd.DataFrame({
        "isic_id": df["isic_id"],
        "target": df[cfg.data.target_col],
        "image_pred": oof_preds,
    })
    oof_path = output_dir / "oof_image_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    logger.info(f"OOF predictions saved → {oof_path}")

    # Overall metrics (only on folds that were actually trained)
    if oof_mask.sum() > 0:
        y_true = df[cfg.data.target_col].values[oof_mask]
        y_pred = oof_preds[oof_mask]
        overall = compute_metrics(y_true, y_pred)
        logger.info(
            f"\nOverall OOF (folds {fold_indices}): "
            f"pAUC={overall['pauc']:.4f} AUC={overall['roc_auc']:.4f}"
        )


if __name__ == "__main__":
    main()
