import os
import math
import numpy as np
import pandas as pd
import cv2
import keras
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, cohen_kappa_score
)
from training.dataset import DatasetManager
from training.preprocessing import FundusPreprocessor

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

def calculate_ece(y_true, y_prob, n_bins=10):
    """Calculates Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = predictions == y_true
    
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i+1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
            
    return ece

def calculate_entropy(probs):
    """Calculates Shannon Entropy (in bits) of probability vector."""
    probs = np.clip(probs, 1e-12, 1.0)
    return -np.sum(probs * np.log2(probs), axis=-1)

def apply_temperature_scaling(logits, temperature):
    """Applies temperature scaling to raw logits and returns calibrated probabilities."""
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

def fit_temperature(logits, y_true):
    """Fits optimal Temperature parameter T on validation set using Negative Log-Likelihood."""
    def nll_loss(t):
        temp = t[0]
        probs = apply_temperature_scaling(logits, temp)
        probs = np.clip(probs, 1e-12, 1.0)
        nll = -np.mean(np.log(probs[np.arange(len(y_true)), y_true]))
        return nll
        
    res = minimize(nll_loss, [1.0], bounds=[(0.01, 10.0)], method='L-BFGS-B')
    return float(res.x[0])

def main():
    print("==================================================")
    print("   PHASE 2-6: VALIDATION SET EVALUATION & CALIBRATION")
    print("==================================================")
    
    os.makedirs("outputs/model_validation", exist_ok=True)
    
    # 1. Load Dataset Splits
    manager = DatasetManager(dataset_dir="aptos2019-blindness-detection")
    metadata_df = manager.load_metadata()
    train_df, val_df, test_df = manager.get_stratified_splits(metadata_df, random_state=42)
    
    print(f"\n[+] Validation Set Count: {len(val_df)} samples")
    print(f"[+] Held-Out Test Set Count: {len(test_df)} samples")
    
    # 2. Load Model & Preprocessor
    model_path = os.path.join("models", "final_efficientnet_b0_dr_model.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join("models", "best_efficientnet_b0_stage1.h5")
        
    print(f"\n[+] Loading Trained Model from: '{model_path}'")
    model = keras.models.load_model(model_path)
    preprocessor = FundusPreprocessor(target_size=(224, 224), apply_crop=True, apply_clahe=True)
    
    # Extract model before final softmax to get raw logits if possible, or invert softmax
    # Reconstruct logits: log(prob)
    
    # 3. Perform Validation Set Inference (Phase 2)
    val_records = []
    y_val_true = []
    y_val_prob = []
    val_logits = []
    
    print(f"\n[+] Running inference on Validation Set ({len(val_df)} samples)...")
    
    for idx, row in val_df.reset_index(drop=True).iterrows():
        id_code = row['id_code']
        true_label = int(row['diagnosis'])
        img_path = row['image_path']
        
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        norm_img = preprocessor.preprocess_image(rgb)
        input_tensor = np.expand_dims(norm_img, axis=0)
        
        probs = model.predict(input_tensor, verbose=0)[0]
        probs_clipped = np.clip(probs, 1e-12, 1.0)
        logits = np.log(probs_clipped)
        
        sorted_indices = np.argsort(probs)[::-1]
        top_idx = sorted_indices[0]
        second_idx = sorted_indices[1]
        
        top_conf = float(probs[top_idx])
        second_conf = float(probs[second_idx])
        margin = top_conf - second_conf
        pred_label = int(top_idx)
        is_correct = bool(pred_label == true_label)
        entropy = float(calculate_entropy(probs))
        
        y_val_true.append(true_label)
        y_val_prob.append(probs)
        val_logits.append(logits)
        
        val_records.append({
            "filename": f"{id_code}.png",
            "actual_class": true_label,
            "predicted_class": pred_label,
            "probability_0": probs[0],
            "probability_1": probs[1],
            "probability_2": probs[2],
            "probability_3": probs[3],
            "probability_4": probs[4],
            "top_confidence": top_conf,
            "second_confidence": second_conf,
            "confidence_margin": margin,
            "entropy": entropy,
            "correct": is_correct
        })
        
    y_val_true = np.array(y_val_true)
    y_val_prob = np.array(y_val_prob)
    val_logits = np.array(val_logits)
    
    # Save prediction_debug.csv
    debug_df = pd.DataFrame(val_records)
    debug_df.to_csv("outputs/model_validation/prediction_debug.csv", index=False)
    print(f"[OK] Saved 'outputs/model_validation/prediction_debug.csv'")
    
    # Save probability_distribution.csv
    dist_df = debug_df[["filename", "actual_class", "predicted_class", "probability_0", "probability_1", "probability_2", "probability_3", "probability_4"]]
    dist_df.to_csv("outputs/model_validation/probability_distribution.csv", index=False)
    print(f"[OK] Saved 'outputs/model_validation/probability_distribution.csv'")
    
    # 4. Phase 3: Metrics Calculation
    y_val_pred = np.argmax(y_val_prob, axis=1)
    acc = accuracy_score(y_val_true, y_val_pred)
    macro_prec = precision_score(y_val_true, y_val_pred, average='macro', zero_division=0)
    macro_rec = recall_score(y_val_true, y_val_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_val_true, y_val_pred, average='macro', zero_division=0)
    
    weight_prec = precision_score(y_val_true, y_val_pred, average='weighted', zero_division=0)
    weight_rec = recall_score(y_val_true, y_val_pred, average='weighted', zero_division=0)
    weight_f1 = f1_score(y_val_true, y_val_pred, average='weighted', zero_division=0)
    
    qwk = cohen_kappa_score(y_val_true, y_val_pred, weights='quadratic')
    
    y_val_onehot = keras.utils.to_categorical(y_val_true, num_classes=5)
    macro_auc = roc_auc_score(y_val_onehot, y_val_prob, average='macro', multi_class='ovr')
    
    per_prec = precision_score(y_val_true, y_val_pred, average=None, zero_division=0)
    per_rec = recall_score(y_val_true, y_val_pred, average=None, zero_division=0)
    per_f1 = f1_score(y_val_true, y_val_pred, average=None, zero_division=0)
    
    # Save classification_report.txt
    report_text = f"""==================================================
RETINAX AI — VALIDATION SET CLASSIFICATION REPORT
==================================================

Total Validation Samples : {len(val_df)}
Overall Accuracy         : {acc*100:.2f}%
Quadratic Weighted Kappa : {qwk:.4f}
Macro ROC-AUC            : {macro_auc*100:.2f}%

Macro Precision          : {macro_prec*100:.2f}%
Macro Recall             : {macro_rec*100:.2f}%
Macro F1-Score           : {macro_f1*100:.2f}%

Weighted Precision       : {weight_prec*100:.2f}%
Weighted Recall          : {weight_rec*100:.2f}%
Weighted F1-Score        : {weight_f1*100:.2f}%

--------------------------------------------------
PER-CLASS DETAILED BREAKDOWN:
--------------------------------------------------
Class 0 (No DR)             - Precision: {per_prec[0]*100:.2f}%, Recall: {per_rec[0]*100:.2f}%, F1: {per_f1[0]*100:.2f}%
Class 1 (Mild)              - Precision: {per_prec[1]*100:.2f}%, Recall: {per_rec[1]*100:.2f}%, F1: {per_f1[1]*100:.2f}%
Class 2 (Moderate)          - Precision: {per_prec[2]*100:.2f}%, Recall: {per_rec[2]*100:.2f}%, F1: {per_f1[2]*100:.2f}%
Class 3 (Severe)            - Precision: {per_prec[3]*100:.2f}%, Recall: {per_rec[3]*100:.2f}%, F1: {per_f1[3]*100:.2f}%
Class 4 (Proliferative DR)  - Precision: {per_prec[4]*100:.2f}%, Recall: {per_prec[4]*100:.2f}%, F1: {per_f1[4]*100:.2f}%
"""
    with open("outputs/model_validation/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[OK] Saved 'outputs/model_validation/classification_report.txt'")
    
    # 5. Phase 4: Confusion Matrix Plot
    cm = confusion_matrix(y_val_true, y_val_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Normalized Confusion Matrix (Validation Set)", fontsize=12, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45)
    plt.yticks(tick_marks, CLASS_NAMES)
    
    thresh = cm_norm.max() / 2.
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, f"{cm_norm[i, j]*100:.1f}%\n({cm[i, j]})",
                     horizontalalignment="center",
                     color="white" if cm_norm[i, j] > thresh else "black",
                     fontsize=9)
                     
    plt.tight_layout()
    plt.ylabel("True Diagnosis Grade")
    plt.xlabel("Predicted Diagnosis Grade")
    plt.savefig("outputs/model_validation/confusion_matrix.png", dpi=300)
    plt.close()
    print(f"[OK] Saved 'outputs/model_validation/confusion_matrix.png'")
    
    # 6. Phase 6: Temperature Scaling Calibration
    optimal_temp = fit_temperature(val_logits, y_val_true)
    y_val_prob_calibrated = apply_temperature_scaling(val_logits, optimal_temp)
    
    ece_before = calculate_ece(y_val_true, y_val_prob)
    ece_after = calculate_ece(y_val_true, y_val_prob_calibrated)
    
    brier_before = float(np.mean(np.sum((y_val_prob - y_val_onehot)**2, axis=1)))
    brier_after = float(np.mean(np.sum((y_val_prob_calibrated - y_val_onehot)**2, axis=1)))
    
    print("\n--------------------------------------------------")
    print("PHASE 6: TEMPERATURE SCALING CALIBRATION RESULTS")
    print("--------------------------------------------------")
    print(f"Optimal Temperature (T)  : {optimal_temp:.4f}")
    print(f"ECE Before Calibration   : {ece_before*100:.2f}%")
    print(f"ECE After Calibration    : {ece_after*100:.2f}%")
    print(f"Brier Score Before       : {brier_before:.4f}")
    print(f"Brier Score After        : {brier_after:.4f}")
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    main()
