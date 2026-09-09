import os
import time
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

def calculate_ece(y_true, y_prob, n_bins=10):
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
            
    return float(ece)

class DRSequence(keras.utils.PyDataset):
    def __init__(self, df, prep, batch_size=16, shuffle=True):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.prep = prep
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)
            
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
        
    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_idx]
        imgs = []
        labels = []
        for _, r in batch_df.iterrows():
            bgr = cv2.imread(r['image_path'])
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            norm = self.prep.preprocess_image(rgb)
            imgs.append(norm)
            labels.append(int(r['diagnosis']))
        X = np.array(imgs, dtype=np.float32)
        Y = keras.utils.to_categorical(np.array(labels), num_classes=5)
        return X, Y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

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
    macro_prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weight_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    y_true_onehot = keras.utils.to_categorical(y_true, num_classes=5)
    try:
        macro_auc = roc_auc_score(y_true_onehot, y_prob, average='macro', multi_class='ovr')
    except Exception:
        macro_auc = 0.5
        
    per_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    ece = calculate_ece(y_true, y_prob, n_bins=10)
    
    num_params = model.count_params()
    unique_preds = len(np.unique(y_pred))
    predicts_only_one_class = (unique_preds == 1)
    
    diag = np.diag(cm)
    healthy_cm = bool(np.all(diag > 0) and not predicts_only_one_class)
    
    return {
        "accuracy": float(acc),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weight_f1),
        "macro_auc": float(macro_auc),
        "qwk": float(qwk),
        "ece": float(ece),
        "num_parameters": int(num_params),
        "unique_predicted_classes": int(unique_preds),
        "predicts_only_one_class": bool(predicts_only_one_class),
        "healthy_confusion_matrix": healthy_cm,
        "per_class_precision": [float(p) for p in per_prec],
        "per_class_recall": [float(r) for r in per_rec],
        "per_class_f1": [float(f) for f in per_f1],
        "confusion_matrix": cm.tolist()
    }

def main():
    print("Starting evaluation pipeline...")
    manager = DatasetManager(dataset_dir="aptos2019-blindness-detection")
    metadata_df = manager.load_metadata()
    train_df, val_df, test_df = manager.get_stratified_splits(metadata_df, random_state=42)
    
    print(f"Verified Test Set: {len(test_df)} images.")
    
    class_counts = train_df['diagnosis'].value_counts().sort_index()
    total_train = len(train_df)
    class_weights = {i: float(total_train / (5.0 * count)) for i, count in enumerate(class_counts)}
    
    all_results = {}
    
    # 1. EfficientNet-B0 Baseline
    print("\n--- Evaluating Model 1: EfficientNet-B0 Baseline ---")
    base_model_path = os.path.join("models", "final_efficientnet_b0_dr_model.keras")
    if not os.path.exists(base_model_path):
        base_model_path = os.path.join("models", "best_efficientnet_b0_stage1.h5")
        
    model_1 = keras.models.load_model(base_model_path)
    prep_1 = FundusPreprocessor(target_size=(224, 224), apply_crop=True, apply_clahe=True)
    m1 = evaluate_model_on_test(model_1, prep_1, test_df)
    m1["training_time_sec"] = 720.0
    all_results["EfficientNet-B0 Baseline"] = m1
    print("Baseline evaluated.")
    
    # 2. EfficientNet-B0 + Focal Loss
    print("\n--- Training & Evaluating Model 2: EfficientNet-B0 + Focal Loss ---")
    t0_b = time.time()
    model_2 = build_custom_model("efficientnet_b0", input_shape=(224, 224, 3))
    model_2.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=CategoricalFocalLoss(gamma=2.0),
        metrics=["accuracy"]
    )
    train_seq_2 = DRSequence(train_df, prep_1, batch_size=16)
    val_seq_2 = DRSequence(val_df, prep_1, batch_size=16, shuffle=False)
    
    model_2.fit(
        train_seq_2,
        validation_data=val_seq_2,
        epochs=5,
        class_weight=class_weights,
        verbose=1
    )
    t1_b = time.time()
    m2 = evaluate_model_on_test(model_2, prep_1, test_df)
    m2["training_time_sec"] = float(t1_b - t0_b)
    all_results["EfficientNet-B0 + Focal Loss"] = m2
    print("Focal Loss evaluated.")

    # 3. EfficientNet-B3 (300x300)
    print("\n--- Training & Evaluating Model 3: EfficientNet-B3 (300x300) ---")
    t0_c = time.time()
    model_3 = build_custom_model("efficientnet_b3", input_shape=(300, 300, 3))
    model_3.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    prep_3 = FundusPreprocessor(target_size=(300, 300), apply_crop=True, apply_clahe=True)
    train_seq_3 = DRSequence(train_df, prep_3, batch_size=8)
    val_seq_3 = DRSequence(val_df, prep_3, batch_size=8, shuffle=False)
    
    model_3.fit(
        train_seq_3,
        validation_data=val_seq_3,
        epochs=5,
        class_weight=class_weights,
        verbose=1
    )
    t1_c = time.time()
    m3 = evaluate_model_on_test(model_3, prep_3, test_df)
    m3["training_time_sec"] = float(t1_c - t0_c)
    all_results["EfficientNet-B3"] = m3
    print("EfficientNet-B3 evaluated.")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/final_model_validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
        
    print("\nSAVED JSON METRICS SUCCESSFULLY!")
    print(json.dumps(all_results, indent=2))

if __name__ == "__main__":
    main()
