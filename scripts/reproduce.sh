#!/usr/bin/env bash
# reproduce.sh — End-to-end reproduction of all three phases.
#
# Prerequisites:
#   - Kaggle data downloaded to data/ (train-metadata.csv, train-image.hdf5)
#   - Phase 1 environment: conda env create -f environment.yml && conda activate isic2024
#   - Phase 2 environment: conda env create -f environment_phase2.yml && conda activate isic2024-phase2
#
# Phase 2 requires a GPU (tested on T4/L40S/H100 via Lightning.ai).
# Phase 1 and Phase 3 run on CPU.
#
# Usage:
#   bash scripts/reproduce.sh          # Run all phases
#   bash scripts/reproduce.sh phase1   # Run Phase 1 only
#   bash scripts/reproduce.sh phase2   # Run Phase 2 only (requires GPU)
#   bash scripts/reproduce.sh phase3   # Run Phase 3 only (requires Phase 1+2 outputs)

set -euo pipefail

PHASE="${1:-all}"

# ── Phase 1: Tabular GBDT Ensemble ──────────────────────────────────────────
run_phase1() {
    echo "═══════════════════════════════════════════════════════════"
    echo "Phase 1: Tabular GBDT Ensemble"
    echo "═══════════════════════════════════════════════════════════"
    python -m isic2024.train --config configs/base.yaml --output-dir outputs
    echo "Phase 1 complete. Outputs: outputs/oof_predictions.csv"
}

# ── Phase 2: Image Deep Learning ────────────────────────────────────────────
run_phase2() {
    echo "═══════════════════════════════════════════════════════════"
    echo "Phase 2: Image Deep Learning (requires GPU)"
    echo "═══════════════════════════════════════════════════════════"

    BACKBONES=(
        "configs/phase2.yaml:outputs/phase2/efficientnetv2_s"
        "configs/phase2_eva02.yaml:outputs/phase2/eva02"
        "configs/phase2_convnextv2.yaml:outputs/phase2/convnextv2"
        "configs/phase2_swin.yaml:outputs/phase2/swin"
    )

    for entry in "${BACKBONES[@]}"; do
        config="${entry%%:*}"
        outdir="${entry##*:}"
        echo "Training: $config -> $outdir"
        python -m isic2024.train_image --config "$config" --output-dir "$outdir" --folds 0,1,2,3,4
    done

    # Optional: SwinV2-B + Tabular conditioning (requires H100)
    echo "Training: SwinV2-B + Tabular conditioning"
    python -m isic2024.train_image \
        --config configs/phase2_swin_tabular.yaml \
        --output-dir outputs/phase2/swin_tabular \
        --folds 0,1,2,3,4

    echo "Phase 2 complete. Outputs: outputs/phase2/*/oof_image_predictions.csv"
}

# ── Phase 3: Stacking Meta-Learner ──────────────────────────────────────────
run_phase3() {
    echo "═══════════════════════════════════════════════════════════"
    echo "Phase 3: Stacking Meta-Learner"
    echo "═══════════════════════════════════════════════════════════"
    python -m isic2024.train_stacking --output-dir outputs/phase3
    echo "Phase 3 complete. Outputs: outputs/phase3/cv_results.json"
}

# ── Dispatch ─────────────────────────────────────────────────────────────────
case "$PHASE" in
    phase1) run_phase1 ;;
    phase2) run_phase2 ;;
    phase3) run_phase3 ;;
    all)
        run_phase1
        run_phase2
        run_phase3
        echo ""
        echo "═══════════════════════════════════════════════════════════"
        echo "All phases complete!"
        echo "═══════════════════════════════════════════════════════════"
        cat outputs/phase3/cv_results.json
        ;;
    *)
        echo "Usage: $0 [phase1|phase2|phase3|all]"
        exit 1
        ;;
esac
