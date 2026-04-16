"""Lightning module for ISIC image classification (Phase 2)."""
from __future__ import annotations

import numpy as np
import timm
import torch
import torch.nn as nn

import lightning as L

from isic2024.config_phase2 import Phase2Config
from isic2024.evaluation.metrics import compute_metrics
from isic2024.models.losses import loss_factory


class ISICImageModule(L.LightningModule):
    """EfficientNet-based binary classifier for skin lesion images."""

    def __init__(self, cfg: Phase2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=["cfg"])

        # Backbone: timm model with no classification head
        self.backbone = timm.create_model(
            cfg.model.backbone,
            pretrained=cfg.model.pretrained,
            num_classes=0,  # feature extractor only
        )
        feat_dim = self.backbone.num_features

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(p=cfg.model.drop_rate),
            nn.Linear(feat_dim, 1),
        )

        self.loss_fn = loss_factory(cfg.loss)

        # Validation accumulators
        self._val_preds: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features).squeeze(-1)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["image"])
        loss = self.loss_fn(logits, batch["target"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        logits = self(batch["image"])
        loss = self.loss_fn(logits, batch["target"])
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self._val_preds.append(logits.detach().cpu())
        self._val_targets.append(batch["target"].detach().cpu())

    def on_validation_epoch_end(self) -> None:
        preds = torch.cat(self._val_preds).sigmoid().numpy()
        targets = torch.cat(self._val_targets).numpy()
        self._val_preds.clear()
        self._val_targets.clear()

        # Guard: need both classes for roc_auc_score (can happen in fast_dev_run)
        n_classes = len(np.unique(targets))
        if n_classes < 2:
            self.log("val/pauc", 0.0, prog_bar=True)
            self.log("val/roc_auc", 0.0)
            self.log("val/brier", 1.0)
            return

        metrics = compute_metrics(targets, preds)
        self.log("val/pauc", metrics["pauc"], prog_bar=True)
        self.log("val/roc_auc", metrics["roc_auc"])
        self.log("val/brier", metrics["brier"])

    def predict_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self(batch["image"]).detach().cpu()

    def configure_optimizers(self) -> dict:
        cfg = self.cfg.optimizer

        # Differential learning rates: backbone vs head
        backbone_params = list(self.backbone.parameters())
        head_params = list(self.head.parameters())

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": cfg.lr_backbone},
                {"params": head_params, "lr": cfg.lr_head},
            ],
            weight_decay=cfg.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[cfg.max_lr_backbone, cfg.max_lr_head],
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=cfg.pct_start,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
