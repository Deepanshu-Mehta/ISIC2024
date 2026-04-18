"""Tests for the Phase 3 stacking meta-learner."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from isic2024.train_stacking import (
    BACKBONES,
    RANK_COLS,
    load_and_merge,
    run_lgbm_stacker,
    run_logreg_stacker,
    run_rank_ensemble,
)


def _create_synthetic_data(tmp_path: Path, n: int = 200, n_pos: int = 10) -> dict[str, Path]:
    """Create synthetic OOF CSVs for testing."""
    rng = np.random.RandomState(42)
    isic_ids = [f"ISIC_{i:07d}" for i in range(n)]
    targets = np.zeros(n, dtype=int)
    targets[:n_pos] = 1
    rng.shuffle(targets)
    folds = np.tile(np.arange(5), n // 5 + 1)[:n]

    # Phase 1 OOF
    phase1_path = tmp_path / "oof_predictions.csv"
    pd.DataFrame({
        "isic_id": isic_ids,
        "target": targets.astype(float),
        "ensemble": rng.rand(n),
    }).to_csv(phase1_path, index=False)

    # Fold assignments
    fold_path = tmp_path / "fold_assignments.csv"
    pd.DataFrame({"isic_id": isic_ids, "fold": folds}).to_csv(fold_path, index=False)

    # Phase 2 backbone OOFs
    phase2_dir = tmp_path / "phase2"
    for backbone in BACKBONES:
        bdir = phase2_dir / backbone
        bdir.mkdir(parents=True)
        pd.DataFrame({
            "isic_id": isic_ids,
            "target": targets,
            "image_pred": rng.rand(n),
        }).to_csv(bdir / "oof_image_predictions.csv", index=False)

    return {
        "phase1_path": phase1_path,
        "phase2_dir": phase2_dir,
        "fold_path": fold_path,
    }


class TestLoadAndMerge:
    def test_correct_shape(self, tmp_path: Path) -> None:
        paths = _create_synthetic_data(tmp_path, n=200, n_pos=10)
        df = load_and_merge(paths["phase1_path"], paths["phase2_dir"], paths["fold_path"])
        assert len(df) == 200
        assert "fold" in df.columns
        for col in RANK_COLS:
            assert col in df.columns

    def test_no_nans(self, tmp_path: Path) -> None:
        paths = _create_synthetic_data(tmp_path, n=100, n_pos=5)
        df = load_and_merge(paths["phase1_path"], paths["phase2_dir"], paths["fold_path"])
        assert df[RANK_COLS].isna().sum().sum() == 0

    def test_rank_values_in_unit_interval(self, tmp_path: Path) -> None:
        paths = _create_synthetic_data(tmp_path, n=100, n_pos=5)
        df = load_and_merge(paths["phase1_path"], paths["phase2_dir"], paths["fold_path"])
        for col in RANK_COLS:
            assert df[col].min() > 0.0
            assert df[col].max() <= 1.0


class TestStackingMethods:
    @pytest.fixture()
    def merged_df(self, tmp_path: Path) -> pd.DataFrame:
        paths = _create_synthetic_data(tmp_path, n=200, n_pos=10)
        return load_and_merge(paths["phase1_path"], paths["phase2_dir"], paths["fold_path"])

    def test_rank_ensemble_returns_valid(self, merged_df: pd.DataFrame) -> None:
        oof, pauc = run_rank_ensemble(merged_df)
        assert len(oof) == len(merged_df)
        assert 0.0 <= pauc <= 0.2
        assert np.all(np.isfinite(oof))

    def test_logreg_stacker_returns_valid(self, merged_df: pd.DataFrame) -> None:
        oof, pauc = run_logreg_stacker(merged_df)
        assert len(oof) == len(merged_df)
        assert 0.0 <= pauc <= 0.2
        assert np.all(np.isfinite(oof))

    def test_lgbm_stacker_returns_valid(self, merged_df: pd.DataFrame) -> None:
        oof, pauc = run_lgbm_stacker(merged_df, n_seeds=1)
        assert len(oof) == len(merged_df)
        assert 0.0 <= pauc <= 0.2
        assert np.all(np.isfinite(oof))
