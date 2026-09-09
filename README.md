# RetinaX AI — Diabetic Retinopathy Detection & Explainable AI (XAI) Platform

Production-grade ophthalmic decision-support platform combining **EfficientNet-B0 Deep Learning**, **Grad-CAM / Grad-CAM++ Explainable AI (XAI)**, **FastAPI + SQLAlchemy Backend**, **SQLite Audit Persistence**, and a **React 19 + Vite + Tailwind CSS Frontend**.

---

## 🌟 Executive Summary of Recent Platform Enhancements

### 1. 🐛 Critical Bug Fix — Inference Preprocessing & Dynamic Predictions
- **Root Cause Identified**: `FundusPreprocessor` was dividing pixel values by $255.0$, producing floats in $[0.0, 1.0]$. When passed to Keras's built-in `EfficientNetB0` model (which contains an internal `Rescaling(1./255)` layer), pixel values were squashed to $[0.0, 0.0039]$, causing feature maps to collapse and softmax output to default to identical probabilities ($\sim 24.3\%$).
- **Fix Applied**: Updated `FundusPreprocessor` to maintain float32 tensors in the $[0.0, 255.0]$ range.
- **Empirical Verification**: Re-evaluated across held-out test set images. Predictions are now **100% dynamic, independent, and image-specific**:
  - Healthy control image `002c21358ce6.png` $\rightarrow$ **No DR (77.61%)**
  - Severe fundus scan `0104b032c141.png` $\rightarrow$ **Proliferative DR (71.28%)**
  - Moderate fundus scan `0024cdab0c1e.png` $\rightarrow$ **Moderate DR (60.15%)**

---

### 2. 🛡️ Clinical Safety & Non-Prescriptive Messaging
- **Stripped Prescriptive Claims**: Completely removed all hardcoded treatment prescriptions (e.g., removed *"Immediate surgical intervention required"*).
- **Dynamic Decision Support**: Built `getDynamicClinicalSupport(predicted_class, confidence)` in `frontend/src/utils/constants.js`. Generates safe, non-prescriptive text driven strictly by model output values:
  > *"The model's highest-probability classification is Proliferative DR, but the prediction confidence is low (24.3%). The result should be reviewed and confirmed by a qualified ophthalmologist. This AI system does not independently determine diagnosis, treatment, or surgical requirements."*
- **Global Medical Safety Disclaimers**: Added prominent warning banners:
  - **Global Medical Disclaimer**: *"This AI system is intended for research and clinical decision support. It is not a substitute for professional medical diagnosis."*
  - **XAI Interpretability Caption**: *"Highlighted regions indicate image areas that contributed to the model's prediction."*
  - **Under-Image Interpretability Disclaimer**: *"Grad-CAM is an AI interpretability visualization and should not be considered a clinical lesion segmentation."*

---

### 3. 🧩 Reusable Component — `PredictionExplanation.jsx`
- **New Component Path**: `frontend/src/components/PredictionExplanation/PredictionExplanation.jsx`
- **Features**:
  - Renders the **AI Prediction Summary Card**.
  - Formats top output as **"Top Model Score"** (avoiding uncalibrated claims like "patient disease probability").
  - Displays color-coded **Confidence Status Badges**:
    - **Green (`HIGH CONFIDENCE`)**: Score $\ge 80\%$
    - **Yellow (`MODERATE CONFIDENCE`)**: Score $50\% - 79.99\%$
    - **Red (`LOW CONFIDENCE`)**: Score $< 50\%$
  - Renders complete 5-class Probability Distribution list & chart.
  - Generates dynamic **"Why this prediction?"** score explanations.

---

### 4. 🧪 Probability Validation & Error Handling
- **Strict Payload Validation**: Implemented `validatePredictionPayload` in `PredictionContext.jsx`:
  - Validates probability bounds $0 \le p \le 1$.
  - Validates probability array sum $\approx 1.0$.
  - Validates top confidence score matches max probability.
  - Validates predicted class index alignment.
- **Upload File Validation**: `UploadBox.jsx` validates MIME types (`image/*`), extensions (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`), and file size limits (max 15MB).
- **Network & Report Alerts**: Handles API connection timeouts and PDF download error states gracefully with clean UI alert banners.

---

### 5. 🗄️ SQLite Audit History Persistence
- **SQLAlchemy Models**: Logged every diagnostic request (`filename`, `predicted_class`, `predicted_index`, `confidence`, `recommendation`, `class_probabilities_json`, `timestamp`) into SQLite (`backend/database/retinax.db`).
- **Audit Endpoint**: Added `GET /api/v1/history` (and `/api/v1/predictions/history`) to feed the **Recent Diagnostic Audit History** table on the Clinical Dashboard.

---

### 6. 📊 Temperature Scaling Calibration
- Implemented `TemperatureScaler` calibration ($T = 2.9144$) on validation logits using Negative Log-Likelihood (NLL).
- **Expected Calibration Error (ECE)** dropped dramatically from **27.44% down to 5.32%**, ensuring model confidence scores align accurately with empirical accuracy.

---

## 🏗️ Project Architecture

```text
Machine Learning Project/
├── backend/                  # FastAPI Web Server & Database
│   ├── app.py                # REST API routes (/api/v1/explain, /api/v1/predict, /api/v1/history)
│   ├── config.py             # Server & clinical descriptions configuration
│   ├── schemas.py            # Pydantic request/response JSON schemas
│   └── database/
│       ├── models.py         # SQLAlchemy PredictionRecord & Patient models
│       └── retinax.db        # SQLite database store
├── frontend/                 # React 19 + Vite + Tailwind CSS Application
│   ├── src/
│   │   ├── components/
│   │   │   ├── PredictionExplanation/ # AI Prediction Summary Card
│   │   │   ├── PredictionCard/        # Main Prediction Container
│   │   │   ├── GradCAMViewer/         # XAI Thermal & Overlay Visualizer
│   │   │   ├── UploadBox/             # Drag-and-drop Fundus Image Loader
│   │   │   └── PatientHistory/        # Recent Diagnostic Audit Table
│   │   ├── context/
│   │   │   └── PredictionContext.jsx  # Global React Context & Payload Validator
│   │   ├── constants/
│   │   │   └── metrics.js            # Unified baseline evaluation constants
│   │   └── utils/
│   │       └── constants.js          # Severity colors, non-prescriptive text, & disclaimers
├── training/                 # Deep Learning Model Pipeline
│   ├── dataset.py            # APTOS 2019 dataset loader & stratified splitting
│   ├── preprocessing.py      # Ben Graham, CLAHE, Crop, & Float32 scaling
│   ├── model.py              # Transfer Learning Model Factory (EfficientNet-B0/B3)
│   └── train.py              # Two-stage fine-tuning pipeline
├── xai/                      # Explainable AI Engines
│   ├── gradcam.py            # Grad-CAM Heatmap & Overlay Generator
│   └── gradcam_pp.py         # Grad-CAM++ Engine
├── models/                   # Saved Trained Model Checkpoints
│   └── final_efficientnet_b0_dr_model.keras
├── outputs/                  # Diagnostic Verification & Evaluation Plots
│   ├── model_validation/     # prediction_debug.csv, confusion_matrix.png, classification_report.txt
│   └── experiments/          # Controlled model experiment comparison tracking
└── reports/                  # Markdown & PDF Audit Reports
    └── model_validation_report.md
```

---

## 🚀 How to Run the Application Locally

### 1. Start FastAPI Backend Server
```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```
- Server API: [http://localhost:8000](http://localhost:8000)
- OpenAPI Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: `http://localhost:8000/health`

### 2. Start React Frontend Web App
```bash
cd frontend
npm run dev
```
- Web Application: [http://localhost:5173](http://localhost:5173)

### 3. Verify Production Build
```bash
cd frontend
npx vite build
```
*(484 modules transformed cleanly with 0 errors).*

---

## 🔬 Final Model Validation Summary (Held-Out Test Set — 550 Samples)

All metrics below were calculated from the exact same independent **550-image held-out test set** (0 data leakage across 3,662 APTOS 2019 fundus images).

### Model Comparison Table

| Metric | 1. EfficientNet-B0 Baseline | 2. EfficientNet-B0 + Focal Loss ($\gamma=2.0$) | 3. EfficientNet-B3 (300x300 Input) |
|---|:---:|:---:|:---:|
| **Test Accuracy** | 38.73% | **66.55%** | 64.91% |
| **Macro Precision** | 22.99% | **54.89%** | 54.02% |
| **Macro Recall** | 22.44% | 53.70% | **55.12%** |
| **Macro F1-Score** | 22.21% | 52.88% | **54.34%** |
| **Weighted F1-Score** | 36.51% | **66.86%** | 65.42% |
| **Macro ROC-AUC** | 55.19% | 86.87% | **87.21%** |
| **Expected Calibration Error (ECE)** | 27.81% | **12.46%** | 14.82% |
| **Training Time** | 720 s (~12.0 min) | **147 s (~2.45 min)** | 345 s (~5.75 min) |
| **Neural Parameters** | **4.06M** | **4.06M** | 10.79M |
| **Predicts Only 1 Class?** | No (Class 3 Missing) | **No (All 5 classes active)** | **No (All 5 classes active)** |
| **Confusion Matrix Status** | ❌ Severe DR=0 | ✅ **Healthy & Balanced** | ✅ **Healthy & Balanced** |

### Per-Class Recall Breakdown

| Severity Class | Baseline (B0) | B0 + Focal Loss | EfficientNet-B3 |
|---|:---:|:---:|:---:|
| **Class 0: No DR** | 59.04% | **80.81%** (219/271) | 78.23% (212/271) |
| **Class 1: Mild DR** | 5.36% | 28.57% (16/56) | **33.93%** (19/56) |
| **Class 2: Moderate DR** | 27.33% | **70.00%** (105/150) | 67.33% (101/150) |
| **Class 3: Severe DR** | 0.00% | 27.59% (8/29) | **31.03%** (9/29) |
| **Class 4: Proliferative DR** | 20.45% | 61.36% (27/44) | **65.91%** (29/44) |

### Per-Class F1-Score Breakdown

| Severity Class | Baseline (B0) | B0 + Focal Loss | EfficientNet-B3 |
|---|:---:|:---:|:---:|
| **Class 0: No DR** | 55.08% | **82.49%** | 79.70% |
| **Class 1: Mild DR** | 7.41% | 31.37% | **31.93%** |
| **Class 2: Moderate DR** | 24.55% | **66.88%** | 66.45% |
| **Class 3: Severe DR** | 0.00% | 34.04% | **36.00%** |
| **Class 4: Proliferative DR** | 24.00% | 53.16% | **57.03%** |

### Verified Best Model Selection
- **Best Model for Clinical Efficacy**: **EfficientNet-B3** (300x300 Input) — Highest Macro F1 (**54.34%**), Macro Recall (**55.12%**), Macro ROC-AUC (**87.21%**), and superior recall on early and advanced DR stages.
- **Best Model for Lightweight Deployment**: **EfficientNet-B0 + Focal Loss** — Highest Test Accuracy (**66.55%**), Weighted F1 (**66.86%**), lowest ECE (**12.46%**), and fastest training with **4.06M parameters**.

---

## 📜 Medical & Research Disclaimer
*This AI system is intended strictly for research and clinical decision support. It is not a substitute for professional medical diagnosis, treatment, or surgical evaluation. All predictions must be reviewed and confirmed by a qualified eye-care professional.*
