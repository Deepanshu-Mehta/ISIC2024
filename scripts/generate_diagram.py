"""Publication-ready architecture diagram for ISIC 2024 hybrid pipeline.

Produces a two-lane figure (ML left, DL right) with:
  - Tensor shapes for each stage
  - Clear ML vs DL colour coding
  - Fusion mechanism detail (rank transform → LogReg)
  - Arrow annotations

Outputs:
    report/figures/architecture.pdf
    report/figures/architecture.png

Usage::

    python scripts/generate_diagram.py
    python scripts/generate_diagram.py --out-dir report/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_ML = "#2171b5"          # Blue — ML / tabular pipeline
C_ML_LIGHT = "#c6dbef"
C_DL = "#d94801"          # Orange-red — DL / image pipeline
C_DL_LIGHT = "#fdd0a2"
C_FUSE = "#31a354"        # Green — fusion / stacking
C_FUSE_LIGHT = "#c7e9c0"
C_DATA = "#756bb1"        # Purple — raw data
C_DATA_LIGHT = "#dadaeb"
C_OUT = "#636363"         # Gray — output
C_OUT_LIGHT = "#d9d9d9"
C_TEXT = "#1a1a1a"

FONT_TITLE = dict(fontsize=9, fontweight="bold", color=C_TEXT, va="center", ha="center")
FONT_SUB = dict(fontsize=7.5, color="#444444", va="center", ha="center")
FONT_SHAPE = dict(fontsize=7, color="#666666", va="center", ha="center",
                  fontstyle="italic")
FONT_ARROW = dict(fontsize=7.5, color="#333333", va="center", ha="center")


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def box(
    ax: plt.Axes,
    cx: float, cy: float,
    w: float, h: float,
    title: str,
    subtitle: str = "",
    shape_str: str = "",
    fc: str = "#ffffff",
    ec: str = "#aaaaaa",
    lw: float = 1.2,
    radius: float = 0.03,
) -> tuple[float, float, float, float]:
    """Draw a rounded rectangle and return (left, right, top, bottom) edges."""
    left, bottom = cx - w / 2, cy - h / 2
    patch = FancyBboxPatch(
        (left, bottom), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        fc=fc, ec=ec, lw=lw, zorder=3,
    )
    ax.add_patch(patch)

    # Title text
    y_title = cy + (0.010 if subtitle else 0)
    ax.text(cx, y_title, title, zorder=4, **FONT_TITLE)

    if subtitle:
        ax.text(cx, cy - 0.020, subtitle, zorder=4, **FONT_SUB)

    if shape_str:
        ax.text(cx, bottom + 0.018, shape_str, zorder=4, **FONT_SHAPE)

    return left, left + w, bottom + h, bottom


def arrow(
    ax: plt.Axes,
    x1: float, y1: float,
    x2: float, y2: float,
    label: str = "",
    color: str = "#555555",
    lw: float = 1.5,
    connectionstyle: str = "arc3,rad=0.0",
) -> None:
    ax.annotate(
        "",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            connectionstyle=connectionstyle,
        ),
        zorder=5,
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.01, my, label, **FONT_ARROW, zorder=6,
                bbox=dict(fc="white", ec="none", pad=1))


def lane_bg(
    ax: plt.Axes,
    x: float, y_top: float, w: float, h: float,
    label: str, fc: str, ec: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y_top - h), w, h,
        boxstyle="round,pad=0,rounding_size=0.02",
        fc=fc, ec=ec, lw=1.5, alpha=0.18, zorder=1,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y_top - 0.022, label,
        fontsize=9.5, fontweight="bold", color=ec,
        va="top", ha="center", zorder=2,
    )


# ---------------------------------------------------------------------------
# Main diagram
# ---------------------------------------------------------------------------

def build_diagram(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ------------------------------------------------------------------
    # Lane backgrounds
    # ------------------------------------------------------------------
    # ML lane: x 0.03 – 0.46, full height
    lane_bg(ax, 0.03, 0.97, 0.43, 0.87, "ML PIPELINE  (Phase 1 — Tabular)",
            C_ML_LIGHT, C_ML)
    # DL lane: x 0.54 – 0.97
    lane_bg(ax, 0.54, 0.97, 0.43, 0.87, "DL PIPELINE  (Phase 2 — Image)",
            C_DL_LIGHT, C_DL)

    # ------------------------------------------------------------------
    # ── ML LANE ──────────────────────────────────────────────────────
    # ------------------------------------------------------------------
    ML_CX = 0.245

    # Raw data
    _, _, top_raw_ml, bot_raw_ml = box(
        ax, ML_CX, 0.895, 0.35, 0.065,
        "train-metadata.csv",
        subtitle="401,059 samples  ·  raw 34 columns",
        shape_str="shape: (401K, 34)",
        fc=C_DATA_LIGHT, ec=C_DATA,
    )

    # Feature engineering
    _, _, top_fe, bot_fe = box(
        ax, ML_CX, 0.775, 0.35, 0.085,
        "Feature Engineering",
        subtitle="color · shape · interaction · location\nugly duckling (patient-wise z-scores)",
        shape_str="→ 88 selected features  (correlation ρ < 0.90, var > 1e−10)",
        fc=C_ML_LIGHT, ec=C_ML,
    )

    # GBDT Ensemble
    _, _, top_gbdt, bot_gbdt = box(
        ax, ML_CX, 0.635, 0.35, 0.095,
        "GBDT Ensemble",
        subtitle="LightGBM + XGBoost + CatBoost\n5 seeds × 5-fold StratifiedGroupKFold\n(grouped by patient_id, seed=42)",
        shape_str="scale_pos_weight=100  ·  early stopping",
        fc=C_ML_LIGHT, ec=C_ML,
    )

    # Rank Ensemble + Calibration
    _, _, top_calib, bot_calib = box(
        ax, ML_CX, 0.505, 0.35, 0.075,
        "Rank Ensemble + Isotonic Calibration",
        subtitle="rank-percentile average of 3 GBDT OOFs\ncalibrate on full OOF (valid: out-of-fold)",
        shape_str="OOF pAUC = 0.1653  ·  AUC = 0.9420",
        fc=C_ML_LIGHT, ec=C_ML,
    )

    # Arrows ML lane
    arrow(ax, ML_CX, bot_raw_ml, ML_CX, top_fe, color=C_DATA)
    arrow(ax, ML_CX, bot_fe, ML_CX, top_gbdt, color=C_ML)
    arrow(ax, ML_CX, bot_gbdt, ML_CX, top_calib, color=C_ML)

    # ------------------------------------------------------------------
    # ── DL LANE ──────────────────────────────────────────────────────
    # ------------------------------------------------------------------
    DL_CX = 0.755

    # Raw data
    _, _, top_raw_dl, bot_raw_dl = box(
        ax, DL_CX, 0.895, 0.35, 0.065,
        "train-images.hdf5",
        subtitle="401,059 JPEG images (decoded on-the-fly)",
        shape_str="shape: (401K,) → decode → (H, W, 3) uint8",
        fc=C_DATA_LIGHT, ec=C_DATA,
    )

    # Augmentation
    _, _, top_aug, bot_aug = box(
        ax, DL_CX, 0.790, 0.35, 0.060,
        "Augmentation  (albumentations)",
        subtitle="RandomFlip · Rotate · ColorJitter · CoarseDropout\nWeightedRandomSampler  neg:pos = 50:1",
        fc=C_DL_LIGHT, ec=C_DL,
    )

    # Backbones
    _, _, top_bb, bot_bb = box(
        ax, DL_CX, 0.668, 0.35, 0.085,
        "4 × timm Backbones",
        subtitle=(
            "SwinV2-B  (87M, 256px)      OOF 0.1549\n"
            "ConvNeXtV2-B  (88M, 224px)  OOF 0.1464\n"
            "EVA02-Small  (22M, 336px)   OOF 0.1403\n"
            "EfficientNetV2-S  (20M, 224px) OOF 0.1399"
        ),
        fc=C_DL_LIGHT, ec=C_DL,
    )

    # Head + TTA
    _, _, top_head, bot_head = box(
        ax, DL_CX, 0.548, 0.35, 0.075,
        "Dropout + Linear Head  →  sigmoid",
        subtitle=(
            "AdamW  ·  OneCycleLR  ·  Focal Loss (γ=2, α=0.25)\n"
            "diff. LR: backbone 1e-4  ·  head 1e-3\n"
            "TTA × 8  (D4 symmetry group, averaged)"
        ),
        fc=C_DL_LIGHT, ec=C_DL,
    )

    # Arrows DL lane
    arrow(ax, DL_CX, bot_raw_dl, DL_CX, top_aug, color=C_DATA)
    arrow(ax, DL_CX, bot_aug, DL_CX, top_bb, color=C_DL)
    arrow(ax, DL_CX, bot_bb, DL_CX, top_head, color=C_DL)

    # ------------------------------------------------------------------
    # ── FUSION ZONE ──────────────────────────────────────────────────
    # ------------------------------------------------------------------
    FUSE_CX = 0.500

    # Rank transform box
    _, _, top_rank, bot_rank = box(
        ax, FUSE_CX, 0.385, 0.55, 0.065,
        "Rank Transform  (5 sources → unified scale)",
        subtitle=(
            "rankdata(pred, method='average') / N   →  [0, 1]\n"
            "rank_tabular  ·  rank_effnet  ·  rank_eva02  ·  rank_convnext  ·  rank_swin"
        ),
        shape_str="shape: (401K, 5)  — all on [0,1] percentile scale",
        fc=C_FUSE_LIGHT, ec=C_FUSE, lw=1.8,
    )

    # LogReg stacker
    _, _, top_lr, bot_lr = box(
        ax, FUSE_CX, 0.265, 0.55, 0.075,
        "Logistic Regression Stacker",
        subtitle=(
            "class_weight='balanced'  ·  C=1.0  ·  solver='lbfgs'\n"
            "5-fold StratifiedGroupKFold CV  (same splits as Phase 1 & 2)\n"
            "Learns differential weights per modality (tabular vs. image)"
        ),
        shape_str="input: (401K, 5)  →  output: (401K,) probabilities",
        fc=C_FUSE_LIGHT, ec=C_FUSE, lw=1.8,
    )

    # Final output
    _, _, top_out, bot_out = box(
        ax, FUSE_CX, 0.135, 0.55, 0.065,
        "Final Prediction",
        subtitle=(
            "OOF pAUC = 0.1745   (+5.6% over best individual model)\n"
            "OOF AUC  = 0.9453"
        ),
        fc=C_OUT_LIGHT, ec=C_OUT, lw=1.8,
    )

    # Arrows: ML + DL lanes → rank transform
    # From ML (tabular calibrated) down-right to rank box
    ax.annotate(
        "",
        xy=(FUSE_CX - 0.22, top_rank),
        xytext=(ML_CX, bot_calib),
        arrowprops=dict(
            arrowstyle="-|>", color=C_ML, lw=1.5,
            connectionstyle="arc3,rad=-0.25",
        ),
        zorder=5,
    )
    ax.text(0.30, 0.452, "tabular OOF\npred → rank", fontsize=7,
            color=C_ML, ha="center", va="center",
            bbox=dict(fc="white", ec="none", pad=1))

    # From DL (head) down-left to rank box
    ax.annotate(
        "",
        xy=(FUSE_CX + 0.22, top_rank),
        xytext=(DL_CX, bot_head),
        arrowprops=dict(
            arrowstyle="-|>", color=C_DL, lw=1.5,
            connectionstyle="arc3,rad=0.25",
        ),
        zorder=5,
    )
    ax.text(0.70, 0.452, "4× image OOF\npred → rank", fontsize=7,
            color=C_DL, ha="center", va="center",
            bbox=dict(fc="white", ec="none", pad=1))

    # Rank → LogReg → Output
    arrow(ax, FUSE_CX, bot_rank, FUSE_CX, top_lr, color=C_FUSE, lw=1.8)
    arrow(ax, FUSE_CX, bot_lr, FUSE_CX, top_out, color=C_FUSE, lw=1.8)

    # ------------------------------------------------------------------
    # ── Legend ───────────────────────────────────────────────────────
    # ------------------------------------------------------------------
    legend_items = [
        mpatches.Patch(fc=C_DATA_LIGHT, ec=C_DATA, lw=1.2, label="Raw Data"),
        mpatches.Patch(fc=C_ML_LIGHT, ec=C_ML, lw=1.2, label="ML Components (Phase 1)"),
        mpatches.Patch(fc=C_DL_LIGHT, ec=C_DL, lw=1.2, label="DL Components (Phase 2)"),
        mpatches.Patch(fc=C_FUSE_LIGHT, ec=C_FUSE, lw=1.8, label="Fusion / Stacking (Phase 3)"),
        mpatches.Patch(fc=C_OUT_LIGHT, ec=C_OUT, lw=1.8, label="Output"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower center",
        ncol=5,
        fontsize=8,
        framealpha=0.9,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, 0.01),
    )

    # ------------------------------------------------------------------
    # ── Title ────────────────────────────────────────────────────────
    # ------------------------------------------------------------------
    fig.suptitle(
        "ISIC 2024 Skin Lesion Classification — Hybrid ML + DL Architecture",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out_path = out_dir / f"architecture.{ext}"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate architecture diagram")
    parser.add_argument("--out-dir", type=Path, default=Path("report/figures"))
    args = parser.parse_args()
    build_diagram(args.out_dir)


if __name__ == "__main__":
    main()
