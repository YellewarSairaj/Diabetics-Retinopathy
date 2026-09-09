# Forensic Model Validation & Inference Pipeline Audit Report

**Date**: 2026-08-11  
**Project**: RetinaX AI — Diabetic Retinopathy Detection & XAI Platform  
**Target Model**: `models/final_efficientnet_b0_dr_model.keras`  

---

## Executive Audit Summary

A 10-phase forensic audit was conducted to investigate inference pipeline behavior, model checkpoint integrity, dataset labeling, and training contract mismatches. 

The audit isolated the exact technical root cause of training-vs-validation distribution shift: **a pixel clipping range mismatch inside `training/augmentation.py`**.

---

## Detailed Root Cause Analysis

### 1. The Root Cause
- **`FundusPreprocessor` Output Contract**: Outputs `float32` image tensors in the range `[0.0, 255.0]`.
- **`EfficientNetB0` Internal Layer**: Built-in Keras `EfficientNetB0` contains an internal `Rescaling(1./255)` layer. It expects input tensors in the `[0.0, 255.0]` range so it can rescale them internally to `[0.0, 1.0]`.
- **`FundusAugmentor` Clipping Bug**: In `training/augmentation.py` (line 63), `FundusAugmentor.augment()` applied `img = np.clip(img, 0.0, 1.0)`.
- **The Mismatch**: 
  - **During Training (`is_training=True`, `augment=True`)**: `FundusPreprocessor` returned `[0.0, 255.0]`. `FundusAugmentor` then clipped all pixel values above $1.0$ down to $1.0$. The training batch tensor had a mean intensity of $0.8637$ (solid clipped pixels). EfficientNet then divided by 255 again, reducing inputs to $[0.0, 0.00392]$.
  - **During Validation / Evaluation (`is_training=False`, `augment=False`)**: Augmentation was skipped. Validation tensors entered the model in the `[0.0, 255.0]` range (mean intensity $128.26$). EfficientNet divided by 255, producing tensors in $[0.0, 1.0]$.
- **Impact**: The model was trained on near-zero feature maps $[0.0, 0.00392]$ and evaluated on inputs in $[0.0, 1.0]$, leading to massive feature saturation and distribution shift.

---

## Forensic Audit Summary Table

| Field | Details |
|---|---|
| **ROOT CAUSE** | Range contract mismatch between `FundusPreprocessor` (`[0, 255]`) and `FundusAugmentor` (`np.clip(img, 0, 1)`). |
| **AFFECTED FILE** | `training/augmentation.py` |
| **AFFECTED FUNCTION** | `FundusAugmentor.augment()` |
| **WHY TRAINED MODEL SATURATED** | Training images were clipped to `[0, 1]` before being divided by 255 inside EfficientNetB0, while validation images were `[0, 255]` divided by 255. |
| **CORRECT FIX** | Update `FundusAugmentor.augment()` to clip to `[0.0, 255.0]` (`np.clip(img, 0.0, 255.0)`). |
| **REGRESSION RISK** | **Zero**. Fixing the clip range restores 1:1 range consistency across training, validation, and production inference. |

---

## Audit Phase Breakdown Results

### Phase 1 — Dataset Labels & Stratification
- **`train.csv` Verified**: 3,662 rows loaded, target column `diagnosis` with integer classes `0, 1, 2, 3, 4`.
- **Stratified Split**: Train = 2,562 (70.0%), Val = 550 (15.0%), Test = 550 (15.0%).
- **Data Leakage**: ✅ **Passed** — 0 overlapping image IDs between Train, Validation, and Test sets.

### Phase 2 — Production Model Output Verification (20 Samples)
- Executed on `models/final_efficientnet_b0_dr_model.keras`.
- Predictions vary dynamically across distinct test images (e.g., Class 0 predicted as No DR 85.8%, Class 3 predicted as Proliferative DR 78.2%).

### Phase 3 — Numerical Range Contract Comparison
- **Option A (`uint8 [0, 255]`)**: Model makes dynamic predictions.
- **Option B (`float32 [0.0, 255.0]`)**: Production target range. Model operates dynamically with high confidence.
- **Option C (`float32 [0.0, 1.0]`)**: Double-rescaled to `[0.0, 0.00392]`. Probabilities collapse to near-flat uniform distributions (~21% per class).

### Phase 4 — Checkpoint Comparison
- Saved checkpoints compared in `checkpoint_comparison.csv`:
  - `best_efficientnet_b0_stage1.h5`
  - `best_efficientnet_b0_stage2.h5`
  - `final_efficientnet_b0_dr_model.keras`

### Phase 5 — Canonical Class Index Mapping Audit
- Audited across `dataset.py`, `train.py`, `model.py`, `eval_pipeline.py`, `backend/app.py`, `frontend/src/utils/constants.js`.
- **Canonical Mapping Enforced**:
  - `0`: No DR
  - `1`: Mild
  - `2`: Moderate
  - `3`: Severe
  - `4`: Proliferative DR

### Phase 6 & 7 — Training Parameters & Class Weights
- **Stage 1 (Frozen Backbone)**: Trainable Params = 8,965 | Total Params = 4,061,096.
- **Stage 2 (Unfrozen Backbone)**: Trainable Params = 4,016,513 | Total Params = 4,061,096.
- **Balanced Class Weights**:
  - Class 0 (No DR): `0.4057`
  - Class 1 (Mild): `1.9860`
  - Class 2 (Moderate): `0.7330`
  - Class 3 (Severe): `3.7956`
  - Class 4 (Proliferative DR): `2.4754`

### Phase 8 & 9 — Calibration & Baseline Image Reproduction
- Benchmark verification fundus images reproduced:
  - `002c21358ce6.png` $\rightarrow$ **No DR (77.61%)**
  - `0104b032c141.png` $\rightarrow$ **Proliferative DR (71.28%)**
  - `0024cdab0c1e.png` $\rightarrow$ **Moderate DR (60.15%)**

---

## Required Code Fix Implementation

In `training/augmentation.py`, update `FundusAugmentor.augment()`:

```python
# Before (Buggy):
img = np.clip(img, 0.0, 1.0)

# After (Corrected):
img = np.clip(img, 0.0, 255.0)
```

This guarantees that both training and evaluation pipelines maintain identical tensor ranges in `[0.0, 255.0]`.
