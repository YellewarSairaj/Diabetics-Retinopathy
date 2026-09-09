import os
import sys
import hashlib
import numpy as np
import pandas as pd
import cv2
import keras
from keras.utils import to_categorical
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, cohen_kappa_score
)

from training.dataset import DatasetManager
from training.preprocessing import FundusPreprocessor

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

def calculate_entropy(probs):
    probs = np.clip(probs, 1e-12, 1.0)
    return -np.sum(probs * np.log2(probs), axis=-1)

def main():
    print("==================================================")
    print("   TASK 1-10: COMPLETE ML PIPELINE AUDIT & TEST")
    print("==================================================")
    
    os.makedirs("outputs/model_validation", exist_ok=True)
    
    # --------------------------------------------------
    # TASK 1: VERIFY TEST DATASET
    # --------------------------------------------------
    print("\n" + "="*50)
    print("TASK 1: VERIFY TEST DATASET")
    print("="*50)
    
    manager = DatasetManager(dataset_dir="aptos2019-blindness-detection")
    metadata_df = manager.load_metadata()
    train_df, val_df, test_df = manager.get_stratified_splits(metadata_df, random_state=42)
    
    print(f"Total Metadata Samples : {len(metadata_df)}")
    print(f"Training Set Size      : {len(train_df)}")
    print(f"Validation Set Size    : {len(val_df)}")
    print(f"Test Set Size          : {len(test_df)}")
    
    test_counts = test_df['diagnosis'].value_counts().sort_index()
    print("\nTest Set Class Distribution:")
    for c in range(5):
        cnt = test_counts.get(c, 0)
        pct = (cnt / len(test_df)) * 100
        print(f"  Class {c} ({CLASS_NAMES[c]:<16}): {cnt:<4} ({pct:.2f}%)")
        
    train_ids = set(train_df['id_code'])
    val_ids = set(val_df['id_code'])
    test_ids = set(test_df['id_code'])
    
    assert len(train_ids.intersection(test_ids)) == 0, "Train/Test overlap!"
    assert len(val_ids.intersection(test_ids)) == 0, "Val/Test overlap!"
    print("[OK] Zero train/test/val filename overlap confirmed.")
    
    # --------------------------------------------------
    # TASK 2: VERIFY MODEL CHECKPOINT
    # --------------------------------------------------
    print("\n" + "="*50)
    print("TASK 2: VERIFY MODEL CHECKPOINT")
    print("="*50)
    
    checkpoint_path = os.path.join("models", "final_efficientnet_b0_dr_model.keras")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join("models", "best_efficientnet_b0_stage1.h5")
        
    print(f"Loading Model Checkpoint: {checkpoint_path}")
    model = keras.models.load_model(checkpoint_path)
    print(f"Model Input Shape  : {model.input_shape}")
    print(f"Model Output Shape : {model.output_shape}")
    print(f"Total Parameters   : {model.count_params():,}")
    
    # --------------------------------------------------
    # TASK 3: VERIFY PREPROCESSING
    # --------------------------------------------------
    print("\n" + "="*50)
    print("TASK 3: VERIFY PREPROCESSING")
    print("="*50)
    preprocessor = FundusPreprocessor(target_size=(224, 224), apply_crop=True, apply_clahe=True)
    print("Preprocessor Config:")
    print(f"  - Target Size: {preprocessor.target_size}")
    print(f"  - Crop Black Borders: {preprocessor.apply_crop}")
    print(f"  - Ben Graham Subtraction: {preprocessor.apply_ben_graham}")
    print(f"  - CLAHE: {preprocessor.apply_clahe}")
    
    # --------------------------------------------------
    # TASK 4 & 5: RUN PREDICTIONS & DIAGNOSTIC CSV
    # --------------------------------------------------
    print("\n" + "="*50)
    print("TASK 4 & 5: RUN PREDICTIONS & DIAGNOSTIC CSV")
    print("="*50)
    
    records = []
    y_true = []
    y_prob = []
    
    for idx, row in test_df.reset_index(drop=True).iterrows():
        id_code = row['id_code']
        filename = f"{id_code}.png"
        true_label = int(row['diagnosis'])
        img_path = row['image_path']
        
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        norm_img = preprocessor.preprocess_image(rgb)
        input_tensor = np.expand_dims(norm_img, axis=0)
        
        probs = model.predict(input_tensor, verbose=0)[0]
        pred_label = int(np.argmax(probs))
        
        sorted_indices = np.argsort(probs)[::-1]
        top_conf = float(probs[sorted_indices[0]])
        second_conf = float(probs[sorted_indices[1]])
        margin = top_conf - second_conf
        is_correct = bool(pred_label == true_label)
        
        y_true.append(true_label)
        y_prob.append(probs)
        
        records.append({
            "filename": filename,
            "true_label": true_label,
            "predicted_label": pred_label,
            "prob_no_dr": probs[0],
            "prob_mild": probs[1],
            "prob_moderate": probs[2],
            "prob_severe": probs[3],
            "prob_pdr": probs[4],
            "top_confidence": top_conf,
            "second_confidence": second_conf,
            "confidence_margin": margin,
            "correct": is_correct
        })
        
    debug_df = pd.DataFrame(records)
    debug_df.to_csv("outputs/model_validation/prediction_debug.csv", index=False)
    print("[OK] Saved 'outputs/model_validation/prediction_debug.csv'")
    
    print("\nFirst 20 Prediction Rows:")
    print(debug_df[["filename", "true_label", "predicted_label", "top_confidence", "correct"]].head(20).to_string())
    
    # Predicted class distribution counts
    pred_counts = debug_df['predicted_label'].value_counts().sort_index()
    print("\nPredicted Class Counts (Test Set):")
    for c in range(5):
        cnt = pred_counts.get(c, 0)
        pct = (cnt / len(debug_df)) * 100
        print(f"  Class {c} ({CLASS_NAMES[c]:<16}): {cnt:<4} ({pct:.2f}%)")
        
    # --------------------------------------------------
    # TASK 6 & 7: RECALCULATE METRICS FROM RAW PREDICTIONS
    # --------------------------------------------------
    print("\n" + "="*50)
    print("TASK 6 & 7: RECALCULATE METRICS FROM RAW PREDICTIONS")
    print("="*50)
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.argmax(y_prob, axis=1)
    
    acc = accuracy_score(y_true, y_pred)
    macro_prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    weight_prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    weight_rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    weight_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    y_true_onehot = to_categorical(y_true, num_classes=5)
    try:
        macro_auc = roc_auc_score(y_true_onehot, y_prob, average='macro', multi_class='ovr')
    except Exception:
        macro_auc = 0.5
        
    per_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    
    print(f"Accuracy                 : {acc*100:.2f}%")
    print(f"Macro F1-Score           : {macro_f1*100:.2f}%")
    print(f"Weighted F1-Score        : {weight_f1*100:.2f}%")
    print(f"Macro ROC-AUC            : {macro_auc*100:.2f}%")
    print(f"Quadratic Weighted Kappa : {qwk:.4f}\n")
    
    print("Confusion Matrix (Labels 0..4):")
    print(cm)
    
    print("\nPer-Class Breakdown:")
    for c in range(5):
        print(f"  Class {c} ({CLASS_NAMES[c]:<16}) - Prec: {per_prec[c]*100:.2f}%, Rec: {per_rec[c]*100:.2f}%, F1: {per_f1[c]*100:.2f}%")

    # --------------------------------------------------
    # TASK 9: COMPARE KNOWN VERIFICATION IMAGES
    # --------------------------------------------------
    print("\n" + "="*50)
    print("TASK 9: COMPARE KNOWN VERIFICATION IMAGES")
    print("="*50)
    known_samples = ["002c21358ce6.png", "0104b032c141.png", "0024cdab0c1e.png"]
    
    for fname in known_samples:
        id_code = fname.replace(".png", "")
        img_path = os.path.join("aptos2019-blindness-detection", "train_images", fname)
        if os.path.exists(img_path):
            bgr = cv2.imread(img_path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            norm = preprocessor.preprocess_image(rgb)
            p = model.predict(np.expand_dims(norm, 0), verbose=0)[0]
            pidx = int(np.argmax(p))
            print(f"Sample '{fname}' -> Predicted: {CLASS_NAMES[pidx]} ({p[pidx]*100:.2f}%) | Probabilities: {np.round(p, 4).tolist()}")

if __name__ == "__main__":
    main()
