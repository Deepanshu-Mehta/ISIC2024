"""HDF5 image dataset and Lightning DataModule for Phase 2."""
from __future__ import annotations

import logging
import warnings
from typing import Any

import albumentations as A
import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import lightning as L

from isic2024.config_phase2 import Phase2Config
from isic2024.data.augmentation import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)


class ISICImageDataset(Dataset):
    """Loads JPEG images from HDF5, decodes on the fly, applies albumentations.

    HDF5 file is opened lazily per worker to avoid fork-safety issues.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        hdf5_path: str,
        transform: A.Compose,
        target_col: str = "target",
    ) -> None:
        self.isic_ids = df["isic_id"].values
        self.targets = df[target_col].values.astype(np.float32)
        self.hdf5_path = hdf5_path
        self.transform = transform
        self._hdf5_file: h5py.File | None = None

    def _open_hdf5(self) -> h5py.File:
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, "r")
        return self._hdf5_file

    def __len__(self) -> int:
        return len(self.isic_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        isic_id = str(self.isic_ids[idx])
        target = self.targets[idx]

        try:
            fp = self._open_hdf5()
            jpeg_bytes = fp[isic_id][()]
            image = cv2.imdecode(
                np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR,
            )
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception:
            warnings.warn(f"Failed to decode image {isic_id}, using black image")
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        augmented = self.transform(image=image)
        return {
            "image": augmented["image"],
            "target": torch.tensor(target, dtype=torch.float32),
            "isic_id": isic_id,
        }


def _worker_init_fn(worker_id: int) -> None:
    """Reset HDF5 file handle in each worker for fork safety."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        if isinstance(dataset, ISICImageDataset):
            dataset._hdf5_file = None


class ISICDataModule(L.LightningDataModule):
    """Lightning DataModule for ISIC image classification."""

    def __init__(
        self,
        cfg: Phase2Config,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_df = train_df
        self.val_df = val_df

    def setup(self, stage: str | None = None) -> None:
        cfg = self.cfg
        train_tfm = get_train_transforms(
            cfg.augment, cfg.image.size,
            cfg.image.normalize_mean, cfg.image.normalize_std,
        )
        val_tfm = get_val_transforms(
            cfg.augment, cfg.image.size,
            cfg.image.normalize_mean, cfg.image.normalize_std,
        )
        self.train_dataset = ISICImageDataset(
            self.train_df, cfg.image.hdf5_path, train_tfm, cfg.data.target_col,
        )
        self.val_dataset = ISICImageDataset(
            self.val_df, cfg.image.hdf5_path, val_tfm, cfg.data.target_col,
        )

    def train_dataloader(self) -> DataLoader:
        cfg = self.cfg
        targets = self.train_dataset.targets
        n_pos = (targets == 1).sum()
        n_neg = (targets == 0).sum()

        samples_per_epoch = int(n_pos * (1 + cfg.sampler.neg_pos_ratio))
        weight_pos = 1.0
        weight_neg = n_pos * cfg.sampler.neg_pos_ratio / max(n_neg, 1)
        weights = np.where(targets == 1, weight_pos, weight_neg)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights).double(),
            num_samples=samples_per_epoch,
            replacement=cfg.sampler.replacement,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=cfg.train.batch_size,
            sampler=sampler,
            num_workers=cfg.train.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            persistent_workers=cfg.train.num_workers > 0,
            worker_init_fn=_worker_init_fn,
        )

    def val_dataloader(self) -> DataLoader:
        cfg = self.cfg
        return DataLoader(
            self.val_dataset,
            batch_size=cfg.train.batch_size * 2,
            shuffle=False,
            num_workers=cfg.train.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=cfg.train.num_workers > 0,
            worker_init_fn=_worker_init_fn,
        )
