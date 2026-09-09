import os
import math
import json
import numpy as np
import pandas as pd
import cv2
import keras
from keras import layers, models
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, cohen_kappa_score
)
from training.dataset import DatasetManager
from training.preprocessing import FundusPreprocessor

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# Focal Loss Implementation for Multi-Class Categorical Imbalance
class CategoricalFocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=None, name="categorical_focal_loss"):
        super().__init__(name=name)
        self.gamma = float(gamma)
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_weight = tf.pow(1.0 - y_pred, self.gamma)
        loss = focal_weight * cross_entropy
        if self.alpha is not None:
            loss = loss * self.alpha
        return tf.reduce_sum(loss, axis=-1)

def build_custom_model(architecture="efficientnet_b0", input_shape=(224, 224, 3), num_classes=5):
    if architecture == "efficientnet_b0":
        backbone = keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=input_shape)
    elif architecture == "efficientnet_b3":
        backbone = keras.applications.EfficientNetB3(include_top=False, weights="imagenet", input_shape=input_shape)
    else:
        raise ValueError(f"Unknown architecture {architecture}")
        
    backbone.trainable = True
    inputs = keras.Input(shape=input_shape, name="input_image")
    features = backbone(inputs)
    x = layers.GlobalAveragePooling2D(name="gap")(features)
    x = layers.BatchNormalization(name="bn")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="top_activation")(x)
    
    return keras.Model(inputs=inputs, outputs=outputs, name=f"DR_{architecture}")

def evaluate_model_on_test(model, preprocessor, test_df):
    y_true = []
    y_prob = []
    
    for idx, row in test_df.reset_index(drop=True).iterrows():
        img_path = row['image_path']
        true_label = int(row['diagnosis'])
        
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        norm_img = preprocessor.preprocess_image(rgb)
        input_tensor = np.expand_dims(norm_img, axis=0)
        
        preds = model.predict(input_tensor, verbose=0)[0]
        y_true.append(true_label)
        y_prob.append(preds)
        
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.argmax(y_prob, axis=1)
    
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weight_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    y_true_onehot = keras.utils.to_categorical(y_true, num_classes=5)
    macro_auc = roc_auc_score(y_true_onehot, y_prob, average='macro', multi_class='ovr')
    
    per_rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weight_f1),
        "macro_auc": float(macro_auc),
        "qwk": float(qwk),
        "grade1_recall": float(per_rec[1]),
        "grade3_recall": float(per_rec[3]),
        "per_class_recall": [float(r) for r in per_rec],
        "per_class_f1": [float(f) for f in per_f1],
        "confusion_matrix": cm.tolist()
    }

def main():
    print("==================================================")
    print("   PHASE 7-8: CONTROLLED MODEL EXPERIMENTS TRACKING")
    print("==================================================")
    
    os.makedirs("outputs/experiments", exist_ok=True)
    
    # 1. Dataset & Splits
    manager = DatasetManager(dataset_dir="aptos2019-blindness-detection")
    metadata_df = manager.load_metadata()
    train_df, val_df, test_df = manager.get_stratified_splits(metadata_df, random_state=42)
    
    # Class weights calculation
    class_counts = train_df['diagnosis'].value_counts().sort_index()
    total_train = len(train_df)
    class_weights = {i: float(total_train / (5.0 * count)) for i, count in enumerate(class_counts)}
    print(f"[+] Computed Class Weights: {class_weights}")
    
    results = []
    
    # --------------------------------------------------
    # EXPERIMENT A: Baseline EfficientNet-B0
    # --------------------------------------------------
    print("\n--------------------------------------------------")
    print("EXPERIMENT A: EfficientNet-B0 Baseline")
    print("--------------------------------------------------")
    base_model_path = os.path.join("models", "final_efficientnet_b0_dr_model.keras")
    if not os.path.exists(base_model_path):
        base_model_path = os.path.join("models", "best_efficientnet_b0_stage1.h5")
        
    model_a = keras.models.load_model(base_model_path)
    prep_a = FundusPreprocessor(target_size=(224, 224), apply_crop=True, apply_clahe=True)
    metrics_a = evaluate_model_on_test(model_a, prep_a, test_df)
    
    exp_a_res = {
        "experiment": "Experiment A (EfficientNet-B0 Baseline)",
        "model": "EfficientNet-B0",
        "input_size": "(224, 224, 3)",
        "loss": "Categorical Cross-Entropy",
        "class_weight": "Standard Inverse Frequency",
        **metrics_a
    }
    results.append(exp_a_res)
    print(f"[OK] Exp A Accuracy: {metrics_a['accuracy']*100:.2f}%, Macro F1: {metrics_a['macro_f1']*100:.2f}%, QWK: {metrics_a['qwk']:.4f}, Grade 1 Rec: {metrics_a['grade1_recall']*100:.2f}%, Grade 3 Rec: {metrics_a['grade3_recall']*100:.2f}%")
    
    # --------------------------------------------------
    # EXPERIMENT B: EfficientNet-B0 + Focal Loss
    # --------------------------------------------------
    print("\n--------------------------------------------------")
    print("EXPERIMENT B: EfficientNet-B0 + Focal Loss (Gamma=2.0)")
    print("--------------------------------------------------")
    model_b = build_custom_model("efficientnet_b0", input_shape=(224, 224, 3))
    model_b.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=CategoricalFocalLoss(gamma=2.0),
        metrics=["accuracy"]
    )
    
    # Fast 2-epoch fine-tune for controlled comparison
    # Generator prep
    def gen_batch(df_sub, prep, batch_size=32, target_res=(224, 224)):
        while True:
            df_shuffled = df_sub.sample(frac=1.0).reset_index(drop=True)
            for i in range(0, len(df_shuffled), batch_size):
                batch_df = df_shuffled.iloc[i:i+batch_size]
                imgs = []
                labels = []
                for _, r in batch_df.iterrows():
                    bgr = cv2.imread(r['image_path'])
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    norm = prep.preprocess_image(rgb)
                    imgs.append(norm)
                    labels.append(int(r['diagnosis']))
                X = np.array(imgs, dtype=np.float32)
                Y = keras.utils.to_categorical(np.array(labels), num_classes=5)
                yield X, Y

    train_gen_b = gen_batch(train_df, prep_a, batch_size=16)
    val_gen_b = gen_batch(val_df, prep_a, batch_size=16)
    
    model_b.fit(
        train_gen_b,
        steps_per_epoch=len(train_df)//16,
        validation_data=val_gen_b,
        validation_steps=len(val_df)//16,
        epochs=3,
        class_weight=class_weights,
        verbose=1
    )
    
    exp_b_path = os.path.join("models", "efficientnet_b0_focal_loss.keras")
    model_b.save(exp_b_path)
    metrics_b = evaluate_model_on_test(model_b, prep_a, test_df)
    exp_b_res = {
        "experiment": "Experiment B (EfficientNet-B0 + Focal Loss)",
        "model": "EfficientNet-B0",
        "input_size": "(224, 224, 3)",
        "loss": "Categorical Focal Loss (gamma=2.0)",
        "class_weight": "Balanced Class Weights",
        **metrics_b
    }
    results.append(exp_b_res)
    print(f"[OK] Exp B Accuracy: {metrics_b['accuracy']*100:.2f}%, Macro F1: {metrics_b['macro_f1']*100:.2f}%, QWK: {metrics_b['qwk']:.4f}, Grade 1 Rec: {metrics_b['grade1_recall']*100:.2f}%, Grade 3 Rec: {metrics_b['grade3_recall']*100:.2f}%")

    # --------------------------------------------------
    # EXPERIMENT C: EfficientNet-B3 (300x300 Resolution)
    # --------------------------------------------------
    print("\n--------------------------------------------------")
    print("EXPERIMENT C: EfficientNet-B3 (300x300 Input Resolution)")
    print("--------------------------------------------------")
    model_c = build_custom_model("efficientnet_b3", input_shape=(300, 300, 3))
    model_c.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    prep_c = FundusPreprocessor(target_size=(300, 300), apply_crop=True, apply_clahe=True)
    train_gen_c = gen_batch(train_df, prep_c, batch_size=8, target_res=(300, 300))
    val_gen_c = gen_batch(val_df, prep_c, batch_size=8, target_res=(300, 300))
    
    model_c.fit(
        train_gen_c,
        steps_per_epoch=len(train_df)//8,
        validation_data=val_gen_c,
        validation_steps=len(val_df)//8,
        epochs=3,
        class_weight=class_weights,
        verbose=1
    )
    
    exp_c_path = os.path.join("models", "efficientnet_b3_300x300.keras")
    model_c.save(exp_c_path)
    metrics_c = evaluate_model_on_test(model_c, prep_c, test_df)
    exp_c_res = {
        "experiment": "Experiment C (EfficientNet-B3 300x300)",
        "model": "EfficientNet-B3",
        "input_size": "(300, 300, 3)",
        "loss": "Categorical Cross-Entropy",
        "class_weight": "Balanced Class Weights",
        **metrics_c
    }
    results.append(exp_c_res)
    print(f"[OK] Exp C Accuracy: {metrics_c['accuracy']*100:.2f}%, Macro F1: {metrics_c['macro_f1']*100:.2f}%, QWK: {metrics_c['qwk']:.4f}, Grade 1 Rec: {metrics_c['grade1_recall']*100:.2f}%, Grade 3 Rec: {metrics_c['grade3_recall']*100:.2f}%")

    # --------------------------------------------------
    # Save Experiment Tracking JSON & Summary CSV
    # --------------------------------------------------
    with open("outputs/experiments/experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    summary_data = []
    for r in results:
        summary_data.append({
            "Model": r["model"],
            "Input Size": r["input_size"],
            "Loss": r["loss"],
            "Class Weight": r["class_weight"],
            "Accuracy": f"{r['accuracy']*100:.2f}%",
            "Macro F1": f"{r['macro_f1']*100:.2f}%",
            "Weighted F1": f"{r['weighted_f1']*100:.2f}%",
            "Macro AUC": f"{r['macro_auc']*100:.2f}%",
            "QWK": f"{r['qwk']:.4f}",
            "Grade 1 Recall": f"{r['grade1_recall']*100:.2f}%",
            "Grade 3 Recall": f"{r['grade3_recall']*100:.2f}%"
        })
        
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("outputs/experiments/experiment_comparison.csv", index=False)
    print("\n[OK] Saved 'outputs/experiments/experiment_comparison.csv'")
    print("==================================================")
    print("      EXPERIMENTS COMPLETED SUCCESSFULLY          ")
    print("==================================================")

if __name__ == "__main__":
    main()
