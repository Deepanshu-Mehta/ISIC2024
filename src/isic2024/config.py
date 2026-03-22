"""Configuration dataclasses for ISIC 2024 pipeline.

All settings live in configs/base.yaml. Load with Config.from_yaml(path).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    raw_dir: str = "data/raw/isic-2024-challenge"
    processed_dir: str = "data/processed"
    train_file: str = "train-metadata.csv"
    test_file: str = "test-metadata.csv"
    target_col: str = "target"
    patient_col: str = "patient_id"
    leakage_cols: list[str] = field(default_factory=lambda: [
        "mel_thick_mm", "mel_mitotic_index",
        "iddx_full", "iddx_1", "iddx_2", "iddx_3", "iddx_4", "iddx_5",
    ])
    meta_cols: list[str] = field(default_factory=lambda: [
        "image_type", "copyright_license",
    ])  # tbp_tile_type KEPT — EDA shows mild malignancy signal (3D:white vs 3D:XP)

    @property
    def train_path(self) -> Path:
        return Path(self.raw_dir) / self.train_file

    @property
    def test_path(self) -> Path:
        return Path(self.raw_dir) / self.test_file


@dataclass
class FeaturesConfig:
    correlation_threshold: float = 0.90
    variance_threshold: float = 1e-10
    quasi_const_threshold: float = 0.995
    use_color: bool = True
    use_shape: bool = True
    use_interaction: bool = True
    use_location: bool = True
    use_ugly_duckling: bool = True


@dataclass
class CVConfig:
    n_splits: int = 5
    seed: int = 42


@dataclass
class LGBMConfig:
    objective: str = "binary"
    metric: str = "auc"
    verbosity: int = -1
    n_estimators: int = 2000
    learning_rate: float = 0.05
    num_leaves: int = 63
    max_depth: int = 6
    min_child_samples: int = 1          # disabled — use weight-based constraint below
    min_sum_hessian_in_leaf: float = 10.0  # weight-based (equiv. to XGB min_child_weight)
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    lambda_l1: float = 0.1
    lambda_l2: float = 0.1
    scale_pos_weight: float = 100.0
    early_stopping_rounds: int = 100


@dataclass
class XGBConfig:
    objective: str = "binary:logistic"
    eval_metric: str = "auc"
    n_estimators: int = 2000
    learning_rate: float = 0.05
    max_depth: int = 6
    min_child_weight: int = 10
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    max_delta_step: int = 1
    scale_pos_weight: float = 100.0
    early_stopping_rounds: int = 100
    verbosity: int = 0


@dataclass
class CatBoostConfig:
    iterations: int = 2000
    learning_rate: float = 0.05
    depth: int = 6
    l2_leaf_reg: float = 3.0
    early_stopping_rounds: int = 100
    verbose: int = 0


@dataclass
class SVMConfig:
    kernel: str = "rbf"
    C: float = 1.0
    gamma: str = "scale"
    probability: bool = True
    class_weight: str = "balanced"


@dataclass
class CalibrationConfig:
    method: str = "isotonic"  # isotonic | platt | temperature


@dataclass
class OptunaConfig:
    n_trials: int = 100
    timeout: int = 3600
    direction: str = "maximize"


@dataclass
class SeedAveragingConfig:
    enabled: bool = True
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456, 789, 2024])


@dataclass
class Config:
    eps: float = 1e-8
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    lgbm: LGBMConfig = field(default_factory=LGBMConfig)
    xgb: XGBConfig = field(default_factory=XGBConfig)
    catboost: CatBoostConfig = field(default_factory=CatBoostConfig)
    svm: SVMConfig = field(default_factory=SVMConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    seed_averaging: SeedAveragingConfig = field(default_factory=SeedAveragingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load configuration from a YAML file."""
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        def _build(dc_cls, data: dict) -> Any:
            fields = {f.name for f in dc_cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
            return dc_cls(**{k: v for k, v in data.items() if k in fields})

        return cls(
            eps=raw.get("eps", 1e-8),
            seed=raw.get("seed", 42),
            data=_build(DataConfig, raw.get("data", {})),
            features=_build(FeaturesConfig, raw.get("features", {})),
            cv=_build(CVConfig, raw.get("cv", {})),
            lgbm=_build(LGBMConfig, raw.get("lgbm", {})),
            xgb=_build(XGBConfig, raw.get("xgb", {})),
            catboost=_build(CatBoostConfig, raw.get("catboost", {})),
            svm=_build(SVMConfig, raw.get("svm", {})),
            calibration=_build(CalibrationConfig, raw.get("calibration", {})),
            optuna=_build(OptunaConfig, raw.get("optuna", {})),
            seed_averaging=_build(SeedAveragingConfig, raw.get("seed_averaging", {})),
        )
