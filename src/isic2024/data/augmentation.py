"""Image augmentation pipelines for Phase 2.

Train: heavy augmentation with D4 symmetries, color jitter, blur, dropout.
Val: resize + normalize only.
TTA: 8 deterministic D4 transforms (4 rotations x 2 flips).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2

from isic2024.config_phase2 import AugmentConfig


class DeterministicD4(ImageOnlyTransform):
    """Apply a specific D4 group element: rot90 k times, optionally hflip."""

    def __init__(self, k: int = 0, hflip: bool = False, p: float = 1.0):
        super().__init__(p=p)
        self.k = k
        self.hflip = hflip

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        if self.hflip:
            img = img[:, ::-1, :].copy()
        if self.k > 0:
            img = np.rot90(img, k=self.k).copy()
        return img

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("k", "hflip")


def get_train_transforms(
    cfg: AugmentConfig,
    image_size: int,
    normalize_mean: tuple[float, ...],
    normalize_std: tuple[float, ...],
) -> A.Compose:
    """Training augmentations: geometric + color + dropout."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=cfg.hflip_p),
        A.VerticalFlip(p=cfg.vflip_p),
        A.RandomRotate90(p=cfg.rotate90_p),
        A.Transpose(p=cfg.transpose_p),
        A.ColorJitter(
            brightness=cfg.brightness_limit,
            contrast=cfg.contrast_limit,
            saturation=cfg.saturation_limit,
            hue=cfg.hue_limit,
            p=cfg.color_jitter_p,
        ),
        A.GaussianBlur(blur_limit=(3, cfg.blur_limit), p=cfg.gaussian_blur_p),
        A.CoarseDropout(
            num_holes_range=(1, cfg.num_holes_max),
            hole_height_range=(1, cfg.hole_height_max),
            hole_width_range=(1, cfg.hole_width_max),
            p=cfg.coarse_dropout_p,
        ),
        A.Normalize(mean=list(normalize_mean), std=list(normalize_std)),
        ToTensorV2(),
    ])


def get_val_transforms(
    cfg: AugmentConfig,
    image_size: int,
    normalize_mean: tuple[float, ...],
    normalize_std: tuple[float, ...],
) -> A.Compose:
    """Validation: resize + normalize only."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=list(normalize_mean), std=list(normalize_std)),
        ToTensorV2(),
    ])


def get_tta_transforms(
    cfg: AugmentConfig,
    image_size: int,
    normalize_mean: tuple[float, ...],
    normalize_std: tuple[float, ...],
) -> list[A.Compose]:
    """8 deterministic D4 transforms for test-time augmentation.

    D4 group: identity, rot90, rot180, rot270, hflip, hflip+rot90, hflip+rot180, hflip+rot270
    """
    transforms = []
    for k in [0, 1, 2, 3]:
        for do_hflip in [False, True]:
            transforms.append(A.Compose([
                A.Resize(image_size, image_size),
                DeterministicD4(k=k, hflip=do_hflip),
                A.Normalize(mean=list(normalize_mean), std=list(normalize_std)),
                ToTensorV2(),
            ]))
    return transforms
