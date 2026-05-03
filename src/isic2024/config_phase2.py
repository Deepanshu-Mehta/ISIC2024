"""Configuration dataclasses for Phase 2 (image-based deep learning).

Load with Phase2Config.from_yaml(path). Reuses CVConfig from Phase 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from isic2024.config import CVConfig


@dataclass
class ImageConfig:
    hdf5_path: str = "data/raw/isic-2024-challenge/train-image.hdf5"
    size: int = 224
    normalize_mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, ...] = (0.229, 0.224, 0.225)


@dataclass
class AugmentConfig:
    hflip_p: float = 0.5
    vflip_p: float = 0.5
    rotate90_p: float = 0.5
    transpose_p: float = 0.5
    color_jitter_p: float = 0.5
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    saturation_limit: float = 0.2
    hue_limit: float = 0.1
    gaussian_blur_p: float = 0.3
    blur_limit: int = 7
    coarse_dropout_p: float = 0.3
    num_holes_max: int = 8
    hole_height_max: int = 32
    hole_width_max: int = 32


@dataclass
class SamplerConfig:
    neg_pos_ratio: int = 50
    replacement: bool = True


@dataclass
class ModelConfig:
    backbone: str = "tf_efficientnetv2_s.in21k_ft_in1k"
    pretrained: bool = True
    drop_rate: float = 0.3
    num_classes: int = 1


@dataclass
class TabularCondConfig:
    """Optional tabular conditioning for the image model.

    When enabled, 20 curated clinical features are embedded through a small
    MLP and concatenated with the backbone image features before the head.
    This creates genuine cross-modal interaction (Hybrid Innovation score 5).
    """

    enabled: bool = False
    features: list = field(default_factory=lambda: [
        # Demographics & anatomy
        "age_approx", "sex", "anatom_site_general",
        # Color channels (TBP system measurements)
        "tbp_lv_A", "tbp_lv_Aext", "tbp_lv_B", "tbp_lv_H", "tbp_lv_Hext", "tbp_lv_L",
        # Color differences (ABCD criteria: color variation)
        "tbp_lv_deltaA", "tbp_lv_deltaB", "tbp_lv_deltaLBnorm", "tbp_lv_Cext",
        # Shape & structure (ABCD criteria: border/asymmetry)
        "tbp_lv_area_perim_ratio", "tbp_lv_symm_2axis", "tbp_lv_stdL",
        # Model confidence scores from ISIC system
        "tbp_lv_nevi_confidence", "tbp_lv_dnn_lesion_confidence",
        # Patient-level context (ugly duckling)
        "n_lesions_patient",
        # Spatial location
        "tbp_lv_y",
    ])
    embed_dim: int = 64     # output dimension of tabular MLP
    hidden_dim: int = 128   # hidden layer of tabular MLP
    dropout: float = 0.1


@dataclass
class LossConfig:
    name: str = "focal"
    gamma: float = 2.0
    alpha: float = 0.25


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 1e-2
    scheduler: str = "onecycle"
    max_lr_backbone: float = 1e-4
    max_lr_head: float = 1e-3
    pct_start: float = 0.1


@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 64
    num_workers: int = 4
    precision: str = "16-mixed"
    accumulate_grad_batches: int = 1
    patience: int = 3
    save_top_k: int = 3
    fast_dev_run: bool = False


@dataclass
class WandbConfig:
    project: str = "isic2024-phase2"
    entity: str | None = None
    enabled: bool = True


@dataclass
class DataConfig:
    raw_dir: str = "data/raw/isic-2024-challenge"
    train_file: str = "train-metadata.csv"
    target_col: str = "target"
    patient_col: str = "patient_id"

    @property
    def train_path(self) -> Path:
        return Path(self.raw_dir) / self.train_file


@dataclass
class Phase2Config:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tabular: TabularCondConfig = field(default_factory=TabularCondConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    cv: CVConfig = field(default_factory=CVConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Phase2Config:
        """Load Phase 2 configuration from a YAML file."""
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        def _build(dc_cls: type, data: dict) -> Any:
            fields = {f.name for f in dc_cls.__dataclass_fields__.values()}
            filtered = {k: v for k, v in data.items() if k in fields}
            # Convert lists to tuples for normalize_mean/std
            if dc_cls is ImageConfig:
                for key in ("normalize_mean", "normalize_std"):
                    if key in filtered and isinstance(filtered[key], list):
                        filtered[key] = tuple(filtered[key])
            return dc_cls(**filtered)

        # TabularCondConfig has a list field — build it directly
        tab_raw = raw.get("tabular", {})
        tabular_cfg = TabularCondConfig(
            enabled=tab_raw.get("enabled", False),
            features=tab_raw.get("features", TabularCondConfig().features),
            embed_dim=tab_raw.get("embed_dim", 64),
            hidden_dim=tab_raw.get("hidden_dim", 128),
            dropout=tab_raw.get("dropout", 0.1),
        )

        return cls(
            seed=raw.get("seed", 42),
            data=_build(DataConfig, raw.get("data", {})),
            image=_build(ImageConfig, raw.get("image", {})),
            augment=_build(AugmentConfig, raw.get("augment", {})),
            sampler=_build(SamplerConfig, raw.get("sampler", {})),
            model=_build(ModelConfig, raw.get("model", {})),
            tabular=tabular_cfg,
            loss=_build(LossConfig, raw.get("loss", {})),
            optimizer=_build(OptimizerConfig, raw.get("optimizer", {})),
            train=_build(TrainConfig, raw.get("train", {})),
            wandb=_build(WandbConfig, raw.get("wandb", {})),
            cv=_build(CVConfig, raw.get("cv", {})),
        )
