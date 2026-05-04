---
title: ISIC 2024 Skin Lesion Classifier
emoji: "\U0001F52C"
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.14.0"
app_file: app.py
pinned: false
license: mit
short_description: 3-phase ensemble for skin cancer detection (pAUC=0.1745)
---

# ISIC 2024 Skin Lesion Classification

Binary classification of skin lesions (malignant vs benign) using a 3-phase ensemble pipeline:

- **Phase 1** — Tabular GBDT ensemble (LightGBM + XGBoost + CatBoost), pAUC = 0.1672
- **Phase 2** — SwinV2-B with cross-modal tabular conditioning, pAUC = 0.1588
- **Phase 3** — LogReg meta-stacker on rank-normalised OOF predictions, **pAUC = 0.1745**

401,059 samples, 393 malignant (0.098%), 5-fold StratifiedGroupKFold, all predictions out-of-fold.

Click any lesion image in the gallery to view model predictions and clinical metadata.
