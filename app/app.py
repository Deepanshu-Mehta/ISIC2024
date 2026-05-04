"""ISIC 2024 Skin Lesion Classification — Gradio Demo.

Gallery of 8 precomputed cases (3 benign, 3 malignant, 2 edge cases)
showcasing OOF predictions from the 3-phase ensemble pipeline.
All predictions are out-of-fold (no data leakage).
"""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

APP_DIR = Path(__file__).parent
GALLERY_JSON = APP_DIR / "gallery" / "gallery.json"

with open(GALLERY_JSON) as f:
    GALLERY_DATA: list[dict] = json.load(f)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TYPE_COLOR = {
    "benign_confident":    "#16a34a",
    "malignant_confident": "#dc2626",
    "edge_case":           "#d97706",
}
TYPE_BG = {
    "benign_confident":    "#f0fdf4",
    "malignant_confident": "#fef2f2",
    "edge_case":           "#fffbeb",
}
TYPE_BADGE = {
    "benign_confident":    "BENIGN",
    "malignant_confident": "MALIGNANT",
    "edge_case":           "EDGE CASE",
}
TYPE_INTERP = {
    "benign_confident":    "All models agree this lesion is benign with high confidence.",
    "malignant_confident": "All models flag this lesion as high-risk. Biopsy confirmed malignancy.",
    "edge_case":           "Target=0 (benign), but models assign high malignancy score — visually ambiguous or atypical morphology.",
}

META_LABELS: dict[str, str] = {
    "age_approx":                   "Age",
    "sex":                          "Sex",
    "anatom_site_general":          "Anatomic Site",
    "clin_size_long_diam_mm":       "Lesion Size",
    "tbp_lv_dnn_lesion_confidence": "Lesion Confidence",
    "tbp_lv_nevi_confidence":       "Nevus Confidence",
    "tbp_lv_norm_border":           "Border Irregularity",
    "tbp_lv_norm_color":            "Color Asymmetry",
    "tbp_lv_symm_2axis":            "Shape Asymmetry",
    "tbp_lv_area_perim_ratio":      "Area/Perimeter Ratio",
}

PRED_LABELS: dict[str, str] = {
    "tabular_gbdt":  "Tabular GBDT Ensemble",
    "swin_tabular":  "SwinV2-B + Tabular Fusion",
    "stacker_score": "LogReg Stacker  (final score)",
    "rank_ensemble": "Rank Ensemble",
}

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_meta(val: object, key: str) -> str:
    if val is None:
        return "—"
    if key in ("tbp_lv_dnn_lesion_confidence", "tbp_lv_nevi_confidence"):
        return f"{float(val):.2f}%"
    if key == "age_approx":
        return f"{int(val)} yrs"
    if key == "clin_size_long_diam_mm":
        return f"{float(val):.2f} mm"
    if isinstance(val, float):
        return f"{val:.3f}"
    return str(val).title()


def _bar_color(pct: float) -> str:
    if pct >= 50:
        return "#dc2626"
    if pct >= 20:
        return "#d97706"
    return "#16a34a"


# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------

def _header_html(case: dict) -> str:
    gtype   = case["gallery_type"]
    color   = TYPE_COLOR[gtype]
    bg      = TYPE_BG[gtype]
    badge   = TYPE_BADGE[gtype]
    label   = case["gallery_label"]
    isic_id = case["isic_id"]
    truth   = "Malignant" if case["target"] == 1 else "Benign"
    interp  = TYPE_INTERP[gtype]
    return f"""
<div style="background:{bg};border:1.5px solid {color}40;border-left:5px solid {color};
            padding:16px 20px;border-radius:8px;margin-bottom:2px;">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
    <span style="background:{color};color:#fff;font-size:11px;font-weight:800;
                 padding:4px 10px;border-radius:4px;letter-spacing:0.08em;">{badge}</span>
    <span style="font-size:18px;font-weight:700;color:#111827;">{label}</span>
  </div>
  <div style="font-size:13px;color:#374151;margin-bottom:8px;">
    <span style="margin-right:20px;">ID: <code style="background:#e5e7eb;padding:2px 6px;
          border-radius:3px;font-size:12px;color:#111827;">{isic_id}</code></span>
    <span>Ground truth: <strong style="color:{color};">{truth}</strong></span>
  </div>
  <div style="font-size:12px;color:#6b7280;font-style:italic;border-top:1px solid {color}30;
              padding-top:8px;">{interp}</div>
</div>"""


def _pred_html(predictions: dict) -> str:
    rows = ""
    for key, label in PRED_LABELS.items():
        val = predictions.get(key)
        is_final = key == "stacker_score"
        if val is None:
            pct, val_str = 0.0, "N/A"
        else:
            pct = float(val) * 100
            val_str = f"{float(val):.4f}"
        color = _bar_color(pct)
        border = f"border:1.5px solid {color}50;" if is_final else ""
        bg     = f"background:{color}08;" if is_final else "background:#f9fafb;"
        rows += f"""
<div style="margin-bottom:12px;padding:10px 12px;border-radius:6px;{bg}{border}">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
    <span style="font-size:13px;font-weight:{'700' if is_final else '500'};color:#111827;">
      {label}{'  ★' if is_final else ''}
    </span>
    <span style="font-size:14px;font-weight:800;color:{color};">{val_str}</span>
  </div>
  <div style="background:#e5e7eb;border-radius:4px;height:10px;overflow:hidden;">
    <div style="background:{color};width:{min(pct,100):.1f}%;height:100%;border-radius:4px;"></div>
  </div>
  <div style="text-align:right;font-size:11px;color:#9ca3af;margin-top:3px;">{pct:.1f}% malignancy risk</div>
</div>"""
    return f'<div style="padding:4px 0;">{rows}</div>'


def _meta_html(metadata: dict) -> str:
    rows = ""
    for i, (key, label) in enumerate(META_LABELS.items()):
        val = metadata.get(key)
        row_bg = "#f9fafb" if i % 2 == 0 else "#ffffff"
        rows += f"""
<tr style="background:{row_bg};">
  <td style="padding:7px 12px;color:#6b7280;font-size:13px;
             white-space:nowrap;border-bottom:1px solid #f3f4f6;">{label}</td>
  <td style="padding:7px 12px;font-size:13px;font-weight:600;color:#111827;
             border-bottom:1px solid #f3f4f6;">{_fmt_meta(val, key)}</td>
</tr>"""
    return (
        '<div style="border:1px solid #e5e7eb;border-radius:8px;overflow:hidden;">'
        f'<table style="width:100%;border-collapse:collapse;">{rows}</table></div>'
    )


# ---------------------------------------------------------------------------
# Gradio callback
# ---------------------------------------------------------------------------

def on_select(evt: gr.SelectData) -> tuple[str, str, str]:
    case = GALLERY_DATA[evt.index]
    return (
        _header_html(case),
        _pred_html(case["predictions"]),
        _meta_html(case["metadata"]),
    )


# ---------------------------------------------------------------------------
# Gallery image list
# ---------------------------------------------------------------------------
gallery_images = [
    (str(APP_DIR / c["image_path"]), c["gallery_label"])
    for c in GALLERY_DATA
]

PLACEHOLDER_HTML = """
<div style="border:2px dashed #e5e7eb;border-radius:8px;padding:32px;text-align:center;
            color:#9ca3af;margin-top:8px;">
  <div style="font-size:32px;margin-bottom:8px;">👈</div>
  <div style="font-size:14px;font-weight:500;">Click any lesion image to inspect</div>
  <div style="font-size:12px;margin-top:4px;">Predictions · Clinical metadata · Risk assessment</div>
</div>"""

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
with gr.Blocks(title="ISIC 2024 Skin Lesion Classifier") as demo:

    gr.HTML("""
<div style="text-align:center;padding:24px 16px 8px;border-bottom:1px solid #e5e7eb;margin-bottom:20px;">
  <h1 style="font-size:26px;font-weight:800;color:#111827;margin:0 0 6px 0;">
    ISIC 2024 — Skin Lesion Classification
  </h1>
  <p style="font-size:14px;color:#6b7280;margin:0;">
    3-phase ensemble &nbsp;·&nbsp; Tabular GBDT + SwinV2-B (cross-modal fusion) + LogReg Stacker
  </p>
  <div style="display:flex;justify-content:center;gap:24px;margin-top:12px;flex-wrap:wrap;">
    <span style="background:#f0fdf4;color:#16a34a;padding:4px 12px;border-radius:20px;
                 font-size:13px;font-weight:600;border:1px solid #bbf7d0;">
      Phase 1 pAUC: 0.1672
    </span>
    <span style="background:#eff6ff;color:#2563eb;padding:4px 12px;border-radius:20px;
                 font-size:13px;font-weight:600;border:1px solid #bfdbfe;">
      Phase 2 pAUC: 0.1588
    </span>
    <span style="background:#fdf4ff;color:#9333ea;padding:4px 12px;border-radius:20px;
                 font-size:13px;font-weight:700;border:1px solid #e9d5ff;">
      Phase 3 pAUC: 0.1745 ★
    </span>
    <span style="background:#f9fafb;color:#374151;padding:4px 12px;border-radius:20px;
                 font-size:13px;border:1px solid #e5e7eb;">
      401K samples · 393 malignant (0.098%)
    </span>
  </div>
</div>""")

    with gr.Row(equal_height=False):

        # ---- Left: gallery ------------------------------------------------
        with gr.Column(scale=5):
            gr.HTML("""
<div style="font-size:14px;font-weight:600;color:#374151;margin-bottom:8px;">
  Curated Cases — click to inspect
</div>
<div style="font-size:12px;color:#6b7280;margin-bottom:10px;">
  <span style="display:inline-block;width:10px;height:10px;background:#16a34a;
               border-radius:2px;margin-right:4px;"></span>Clearly Benign &nbsp;
  <span style="display:inline-block;width:10px;height:10px;background:#dc2626;
               border-radius:2px;margin-right:4px;"></span>Confirmed Malignant &nbsp;
  <span style="display:inline-block;width:10px;height:10px;background:#d97706;
               border-radius:2px;margin-right:4px;"></span>Diagnostic Edge Case
</div>""")
            gallery = gr.Gallery(
                value=gallery_images,
                columns=2,
                rows=4,
                height=580,
                object_fit="cover",
                allow_preview=False,
                show_label=False,
            )
            gr.HTML("""
<div style="margin-top:10px;padding:10px 14px;background:#f9fafb;border-radius:6px;
            border:1px solid #e5e7eb;font-size:12px;color:#6b7280;line-height:1.7;">
  All predictions are <strong style="color:#374151;">out-of-fold (OOF)</strong> —
  each sample was predicted by a model that never saw it during training.
  Metric: <strong style="color:#374151;">pAUC above 80% TPR</strong>.
</div>""")

        # ---- Right: detail panel ------------------------------------------
        with gr.Column(scale=5):
            header_box = gr.HTML(value=PLACEHOLDER_HTML)

            gr.HTML('<div style="font-size:14px;font-weight:700;color:#374151;'
                    'margin:14px 0 6px;">Model Predictions</div>')
            pred_box = gr.HTML()

            gr.HTML('<div style="font-size:14px;font-weight:700;color:#374151;'
                    'margin:14px 0 6px;">Clinical Metadata</div>')
            meta_box = gr.HTML()

    gallery.select(on_select, outputs=[header_box, pred_box, meta_box])

    gr.HTML("""
<div style="margin-top:24px;padding:14px 20px;background:#f9fafb;border-radius:8px;
            border:1px solid #e5e7eb;font-size:12px;color:#6b7280;line-height:1.8;">
  <strong style="color:#374151;">Pipeline:</strong>
  Raw CSV (401K × 55) → Preprocessor + Feature Engineering (88 features) →
  LightGBM + XGBoost + CatBoost (5-fold, seed-avg) → RankEnsemble → Isotonic calibration
  &nbsp;+&nbsp;
  HDF5 images → SwinV2-B + 20-feature MLP fusion → focal loss + AdamW + OneCycleLR → TTA×8
  &nbsp;→&nbsp;
  <strong style="color:#374151;">LogReg stacker on 5 rank-normalised OOF sources → pAUC = 0.1745</strong>
  &nbsp;|&nbsp;
  <a href="https://github.com/Deepanshu-Mehta/ISIC2024" style="color:#2563eb;">GitHub</a>
</div>""")

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
