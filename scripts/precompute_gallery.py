"""Precompute gallery data for the Gradio web UI demo.

Selects 8 representative ISIC samples (3 benign, 3 malignant, 2 edge cases),
extracts their JPEG images from HDF5, and saves a gallery JSON with all
predictions and metadata. Run this on Lightning.ai before closing the session.

Outputs (under --gallery-dir):
    images/<isic_id>.jpg     — extracted skin lesion images
    gallery.json             — metadata + all OOF predictions per sample

Usage::

    python scripts/precompute_gallery.py \\
        --config configs/phase2_swin_tabular.yaml \\
        --output-dir outputs/phase2/swin_tabular \\
        --gallery-dir app/gallery
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# Metadata columns to include in gallery JSON (human-readable context)
META_COLS = [
    "isic_id", "target", "age_approx", "sex", "anatom_site_general",
    "clin_size_long_diam_mm", "tbp_lv_dnn_lesion_confidence",
    "tbp_lv_nevi_confidence", "tbp_lv_norm_border", "tbp_lv_norm_color",
    "tbp_lv_symm_2axis", "tbp_lv_area_perim_ratio",
]

CASE_LABELS = {
    "benign_confident": "Clearly Benign",
    "malignant_confident": "Confirmed Malignant",
    "edge_case": "Diagnostic Edge Case",
}


def select_gallery_cases(df_merged: pd.DataFrame) -> list[dict]:
    """Select 8 diverse gallery cases based on stacking predictions.

    Strategy:
      - 3 clearly benign: target=0, very low logreg_stacker score
      - 3 confirmed malignant: target=1, highest logreg_stacker score
      - 2 edge cases: target=0 but high stacker score (model uncertain /
        visually ambiguous)
    """
    cases = []

    # 3 clearly benign — low stacker score, pick variety of sites
    benign = df_merged[df_merged["target"] == 0].copy()
    benign_sorted = benign.nsmallest(500, "logreg_stacker")
    sites = benign_sorted["anatom_site_general"].dropna().unique()
    seen_sites: set = set()
    for _, row in benign_sorted.iterrows():
        site = row.get("anatom_site_general", "unknown")
        if site not in seen_sites:
            cases.append({**row.to_dict(), "_gallery_type": "benign_confident"})
            seen_sites.add(site)
        if len([c for c in cases if c["_gallery_type"] == "benign_confident"]) >= 3:
            break
    # Fallback if not enough site diversity
    while len([c for c in cases if c["_gallery_type"] == "benign_confident"]) < 3:
        row = benign_sorted.iloc[len(cases)]
        cases.append({**row.to_dict(), "_gallery_type": "benign_confident"})

    # 3 confirmed malignant — highest stacker scores
    malignant = df_merged[df_merged["target"] == 1].copy()
    malignant_sorted = malignant.nlargest(10, "logreg_stacker")
    for _, row in malignant_sorted.iterrows():
        cases.append({**row.to_dict(), "_gallery_type": "malignant_confident"})
        if len([c for c in cases if c["_gallery_type"] == "malignant_confident"]) >= 3:
            break

    # 2 edge cases — target=0 but stacker assigns high malignancy score.
    # Most interesting for the demo: model raises an alarm on a benign lesion,
    # prompting the user to think about why (atypical morphology, patient age, etc.)
    selected_ids = {c["isic_id"] for c in cases}
    edge = df_merged[(df_merged["target"] == 0) &
                     (~df_merged["isic_id"].isin(selected_ids))].copy()
    edge_sorted = edge.nlargest(50, "logreg_stacker")
    for _, row in edge_sorted.iterrows():
        cases.append({**row.to_dict(), "_gallery_type": "edge_case"})
        if len([c for c in cases if c["_gallery_type"] == "edge_case"]) >= 2:
            break

    return cases


def extract_image(hdf5_path: str, isic_id: str, out_path: Path) -> bool:
    """Extract JPEG bytes from HDF5 and save as a .jpg file."""
    try:
        with h5py.File(hdf5_path, "r") as f:
            jpeg_bytes = f[isic_id][()]
        out_path.write_bytes(bytes(jpeg_bytes))
        return True
    except Exception as e:
        print(f"  WARNING: could not extract {isic_id}: {e}")
        return False


def safe_val(v) -> object:
    """Convert numpy scalars to Python natives for JSON serialisation."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute Gradio gallery data")
    parser.add_argument("--config", type=str,
                        default="configs/phase2_swin_tabular.yaml")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/phase2/swin_tabular"))
    parser.add_argument("--gallery-dir", type=Path, default=Path("app/gallery"))
    parser.add_argument("--phase1-oof", type=Path,
                        default=Path("outputs/oof_predictions.csv"))
    parser.add_argument("--stacking-oof", type=Path,
                        default=Path("outputs/phase3/oof_stacking_predictions.csv"))
    parser.add_argument("--metadata", type=Path,
                        default=Path("data/raw/isic-2024-challenge/train-metadata.csv"))
    parser.add_argument("--hdf5", type=str,
                        default="data/raw/isic-2024-challenge/train-image.hdf5")
    args = parser.parse_args()

    img_dir = args.gallery_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print("Loading OOF prediction CSVs...")
    df_tab = pd.read_csv(args.phase1_oof, usecols=["isic_id", "ensemble"])
    df_stack = pd.read_csv(args.stacking_oof)
    df_meta = pd.read_csv(args.metadata, low_memory=False)

    # Load swin_tabular OOF
    swin_tab_oof = args.output_dir / "oof_image_predictions.csv"
    df_swin_tab = pd.read_csv(swin_tab_oof, usecols=["isic_id", "image_pred"])
    df_swin_tab = df_swin_tab.rename(columns={"image_pred": "pred_swin_tabular"})

    # Merge everything
    df = df_stack.merge(df_tab.rename(columns={"ensemble": "pred_tabular"}),
                        on="isic_id", how="inner")
    df = df.merge(df_swin_tab, on="isic_id", how="left")

    # Add rank_swin column for edge case selection (from stacking OOF)
    # rank_swin already exists in df_stack
    available_meta = [c for c in META_COLS if c in df_meta.columns]
    df = df.merge(df_meta[available_meta], on="isic_id", how="left",
                  suffixes=("", "_meta"))

    print(f"Merged dataset: {len(df):,} rows, {int(df['target'].sum())} positives")

    print("Selecting 8 gallery cases...")
    cases = select_gallery_cases(df)
    print(f"  Selected: {len(cases)} cases")
    for c in cases:
        print(f"    [{c['_gallery_type']:22s}] {c['isic_id']}  "
              f"target={int(c['target'])}  stacker={c['logreg_stacker']:.4f}")

    print(f"\nExtracting images from {args.hdf5}...")
    gallery = []
    for case in cases:
        isic_id = case["isic_id"]
        img_path = img_dir / f"{isic_id}.jpg"
        ok = extract_image(args.hdf5, isic_id, img_path)

        entry = {
            "isic_id": isic_id,
            "gallery_type": case["_gallery_type"],
            "gallery_label": CASE_LABELS[case["_gallery_type"]],
            "image_path": str(img_path.relative_to(args.gallery_dir.parent)),
            "target": int(case["target"]),
            "image_extracted": ok,
            # Model predictions (all OOF — honest evaluation)
            "predictions": {
                "tabular_gbdt":    safe_val(case.get("pred_tabular")),
                "swin_image":      safe_val(case.get("rank_swin")),
                "swin_tabular":    safe_val(case.get("pred_swin_tabular")),
                "stacker_score":   safe_val(case.get("logreg_stacker")),
                "rank_ensemble":   safe_val(case.get("rank_ensemble")),
            },
            # Clinical metadata for display
            "metadata": {
                col: safe_val(case.get(col))
                for col in META_COLS
                if col not in ("isic_id", "target")
            },
        }
        gallery.append(entry)
        print(f"  {isic_id}: image={'OK' if ok else 'FAILED'}")

    out_json = args.gallery_dir / "gallery.json"
    out_json.write_text(json.dumps(gallery, indent=2))
    print(f"\nSaved: {out_json}")
    print(f"Saved: {len(list(img_dir.glob('*.jpg')))} images in {img_dir}")
    print("\nDone. Copy app/gallery/ back to your local machine.")


if __name__ == "__main__":
    main()
