"""
Optimized System Verification Script for RetinaX AI Platform
============================================================
"""

import os
import sys

# Ensure UTF-8 output encoding for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "training"))
sys.path.insert(0, os.path.join(ROOT_DIR, "xai"))
sys.path.insert(0, os.path.join(ROOT_DIR, "backend"))
sys.path.insert(0, os.path.join(ROOT_DIR, "database"))
sys.path.insert(0, os.path.join(ROOT_DIR, "reports"))

from dataset import DatasetManager, CLASS_NAMES
from preprocessing import FundusPreprocessor
from augmentation import FundusDataSequence
from xai.gradcam import GradCAM
from xai.gradcam_plus_plus import GradCAMPlusPlus

OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("       RETINAX AI PLATFORM — FAST VERIFICATION SUITE       ")
print("="*70)

# ------------------------------------------------------------------------------
# PHASE 1: DATASET VERIFICATION
# ------------------------------------------------------------------------------
print("\n[+] PHASE 1: DATASET VERIFICATION & STATISTICS")
manager = DatasetManager(dataset_dir="aptos2019-blindness-detection")
df = manager.load_metadata()

missing_count = sum(not os.path.exists(row['image_path']) for _, row in df.iterrows())
print(f"    - Total Metadata Records : {len(df)}")
print(f"    - Missing Image Files    : {missing_count}")

dist = df['diagnosis'].value_counts().sort_index()
stats_rows = []
for cid in range(5):
    cnt = int(dist.get(cid, 0))
    pct = (cnt / len(df)) * 100
    stats_rows.append({
        "Class_ID": cid,
        "Class_Name": CLASS_NAMES[cid],
        "Sample_Count": cnt,
        "Percentage": round(pct, 2)
    })

stats_df = pd.DataFrame(stats_rows)
stats_csv = os.path.join(OUTPUT_DIR, "dataset_statistics.csv")
stats_df.to_csv(stats_csv, index=False)
print(f"    [OK] Exported: {stats_csv}")

plt.figure(figsize=(8, 4))
colors = ['#22c55e', '#3b82f6', '#f59e0b', '#f97316', '#ef4444']
bars = plt.bar(stats_df['Class_Name'], stats_df['Sample_Count'], color=colors, edgecolor='black', alpha=0.85)
plt.title('APTOS 2019 Diabetic Retinopathy Class Distribution', fontsize=12, fontweight='bold')
plt.xlabel('DR Severity Grade')
plt.ylabel('Number of Images')
plt.grid(axis='y', linestyle='--', alpha=0.5)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 15, f"{yval}\n({yval/len(df)*100:.1f}%)", ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
dist_img = os.path.join(OUTPUT_DIR, "class_distribution.png")
plt.savefig(dist_img, dpi=300)
plt.close()
print(f"    [OK] Saved Plot: {dist_img}")

# ------------------------------------------------------------------------------
# PHASE 2: PREPROCESSING VERIFICATION
# ------------------------------------------------------------------------------
print("\n[+] PHASE 2: PREPROCESSING PIPELINE VERIFICATION")
preprocessor = FundusPreprocessor(target_size=(224, 224))
sample_img_path = df.iloc[0]['image_path']
raw_bgr = cv2.imread(sample_img_path)
raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

cropped_rgb = preprocessor.crop_black_borders(raw_rgb)
ben_graham_rgb = preprocessor.apply_ben_graham_technique(cropped_rgb)
final_processed = preprocessor.preprocess_image(sample_img_path)

plt.figure(figsize=(12, 3.5))
plt.subplot(1, 3, 1)
plt.imshow(raw_rgb)
plt.title("Original Fundus")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(ben_graham_rgb)
plt.title("Ben Graham Subtraction")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(final_processed)
plt.title("Processed (224x224)")
plt.axis('off')

plt.tight_layout()
prep_img = os.path.join(OUTPUT_DIR, "preprocessing_comparison.png")
plt.savefig(prep_img, dpi=300)
plt.close()
print(f"    [OK] Saved Plot: {prep_img}")

# ------------------------------------------------------------------------------
# PHASE 3: MODEL ARCHITECTURE VALIDATION
# ------------------------------------------------------------------------------
print("\n[+] PHASE 3: MODEL ARCHITECTURE VALIDATION")
model_path = os.path.join(ROOT_DIR, "models", "best_efficientnet_b0_stage1.h5")
if not os.path.exists(model_path):
    model_path = os.path.join(ROOT_DIR, "models", "final_efficientnet_b0_dr_model.keras")

model = keras.models.load_model(model_path)
print(f"    - Active Model     : {os.path.basename(model_path)}")
print(f"    - Input Shape      : {model.input_shape}")
print(f"    - Output Shape     : {model.output_shape}")
print(f"    - Total Parameters : {model.count_params():,}")
assert model.input_shape == (None, 224, 224, 3), "Invalid input shape!"
assert model.output_shape == (None, 5), "Invalid output shape!"
print("    [OK] Shapes (224,224,3) -> (5,) verified.")

# ------------------------------------------------------------------------------
# PHASE 4: PERFORMANCE EVALUATION & ARTIFACT GENERATION
# ------------------------------------------------------------------------------
print("\n[+] PHASE 4: PERFORMANCE EVALUATION")
_, _, test_df = manager.get_stratified_splits(df)
test_sample = test_df.iloc[:50].reset_index(drop=True)
test_gen = FundusDataSequence(test_sample, batch_size=10, is_training=False, shuffle=False, augment=False)

y_proba = model.predict(test_gen, verbose=0)
y_pred = np.argmax(y_proba, axis=1)
y_true = test_sample['diagnosis'].values[:len(y_pred)]

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

metrics_df = pd.DataFrame([{
    "Accuracy": round(acc, 4),
    "Precision": round(prec, 4),
    "Recall": round(rec, 4),
    "F1_Score": round(f1, 4)
}])
metrics_csv = os.path.join(OUTPUT_DIR, "performance_metrics.csv")
metrics_df.to_csv(metrics_csv, index=False)
print(f"    - Accuracy: {acc*100:.2f}% | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
print(f"    [OK] Exported: {metrics_csv}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(CLASS_NAMES.values()), yticklabels=list(CLASS_NAMES.values()))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
cm_plot = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_plot, dpi=300)
plt.close()
print(f"    [OK] Saved Plot: {cm_plot}")

# ROC & PR Curves
y_true_onehot = np.eye(5)[y_true]
plt.figure(figsize=(6, 5))
for i in range(5):
    if len(np.unique(y_true_onehot[:, i])) > 1:
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_proba[:, i])
        score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{CLASS_NAMES[i]} (AUC={score:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curves')
plt.legend()
plt.tight_layout()
roc_plot = os.path.join(OUTPUT_DIR, "roc_curve.png")
plt.savefig(roc_plot, dpi=300)
plt.close()
print(f"    [OK] Saved Plot: {roc_plot}")

plt.figure(figsize=(6, 5))
for i in range(5):
    if len(np.unique(y_true_onehot[:, i])) > 1:
        p_vals, r_vals, _ = precision_recall_curve(y_true_onehot[:, i], y_proba[:, i])
        plt.plot(r_vals, p_vals, label=f"{CLASS_NAMES[i]}")

plt.title('Precision-Recall Curves')
plt.legend()
plt.tight_layout()
pr_plot = os.path.join(OUTPUT_DIR, "pr_curve.png")
plt.savefig(pr_plot, dpi=300)
plt.close()
print(f"    [OK] Saved Plot: {pr_plot}")

# ------------------------------------------------------------------------------
# PHASE 5: GRAD-CAM EXPLAINABILITY
# ------------------------------------------------------------------------------
print("\n[+] PHASE 5: EXPLAINABLE AI VERIFICATION")
gradcam = GradCAM(model=model, layer_name="top_activation")
gradcam_pp = GradCAMPlusPlus(model=model, layer_name="top_activation")

processed_sample = preprocessor.preprocess_image(sample_img_path)
input_tensor = np.expand_dims(processed_sample, axis=0)

heatmap_gc, p_idx, p_conf = gradcam.compute_heatmap(input_tensor)
heatmap_gcpp, _, _ = gradcam_pp.compute_heatmap(input_tensor)

_, overlay_gc = gradcam.overlay_heatmap(heatmap_gc, raw_rgb)
_, overlay_gcpp = gradcam_pp.overlay_heatmap(heatmap_gcpp, raw_rgb)

plt.figure(figsize=(12, 3.5))
plt.subplot(1, 3, 1)
plt.imshow(raw_rgb)
plt.title(f"Original ({CLASS_NAMES[p_idx]})")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(overlay_gc)
plt.title("Grad-CAM Overlay")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(overlay_gcpp)
plt.title("Grad-CAM++ Overlay")
plt.axis('off')

plt.tight_layout()
xai_plot = os.path.join(OUTPUT_DIR, "gradcam_sample_overlay.png")
plt.savefig(xai_plot, dpi=300)
plt.close()
print(f"    [OK] Saved Plot: {xai_plot}")

print("\n" + "="*70)
print("     ALL VERIFICATION PHASES COMPLETED & ARTIFACTS GENERATED     ")
print("="*70)
