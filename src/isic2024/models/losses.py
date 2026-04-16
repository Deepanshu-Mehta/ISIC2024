"""Loss functions for Phase 2 image models."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from isic2024.config_phase2 import LossConfig


class FocalLoss(nn.Module):
    """Binary focal loss operating on raw logits.

    Focal loss down-weights easy examples so the model focuses on hard ones.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)  # p_t = sigmoid(logit) if target=1, else 1-sigmoid(logit)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


def loss_factory(cfg: LossConfig) -> nn.Module:
    """Create a loss function from config."""
    if cfg.name == "focal":
        return FocalLoss(gamma=cfg.gamma, alpha=cfg.alpha)
    raise ValueError(f"Unknown loss: {cfg.name}")
