"""
Grad-CAM Batch Explanation Script
===================================
Domain: Healthcare / Computer Vision / Deep Learning
Dataset: APTOS 2019 Blindness Detection
Task: Run Grad-CAM visual explanation pipeline on sample test images
      and save clinical explanation report dashboards to reports/xai/

Author: Senior AI Engineer & XAI Specialist
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import keras

# Add project root and training dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training"))

from dataset import DatasetManager
from preprocessing import FundusPreprocessor
from xai.gradcam import GradCAM

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]


def run_xai_pipeline(num_samples: int = 5):
    """
    Selects sample fundus images covering various DR severity grades,
    runs inference and Grad-CAM visual explanation, and saves 4-panel reports.
    """
    print("[+] Initializing Grad-CAM Visual Explanation Pipeline...")

    # Load dataset splits
    manager = DatasetManager(dataset_dir="aptos2019-blindness-detection")
    df = manager.load_metadata()
    _, _, test_df = manager.get_stratified_splits(df)

    # Path to trained model
    model_path = os.path.join("models", "final_efficientnet_b0_dr_model.keras")
    if not os.path.exists(model_path):
        # Fallback to checkpoint if final model isn't saved yet
        model_path = os.path.join("models", "best_efficientnet_b0_stage2.h5")
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Trained model file not found in 'models/'. Please run model training first.")
        return

    print(f"[+] Loading model from: '{model_path}'...")
    model = keras.models.load_model(model_path)
    preprocessor = FundusPreprocessor(target_size=(224, 224), apply_crop=True, apply_clahe=True)

    # Initialize GradCAM helper
    try:
        gradcam = GradCAM(model=model, layer_name="top_activation")
    except Exception as e:
        print(f"[WARN] Failed to find layer 'top_activation', searching automatically: {e}")
        gradcam = GradCAM(model=model)

    reports_xai_dir = os.path.join("reports", "xai")
    os.makedirs(reports_xai_dir, exist_ok=True)

    # Select samples (one from each class if available)
    sampled_indices = []
    for cls in range(5):
        cls_samples = test_df[test_df['diagnosis'] == cls]
        if not cls_samples.empty:
            sampled_indices.append(cls_samples.index[0])

    if len(sampled_indices) < num_samples:
        additional = test_df.index.difference(sampled_indices)[:num_samples - len(sampled_indices)]
        sampled_indices.extend(list(additional))

    print(f"\n[+] Processing {len(sampled_indices)} sample fundus images for XAI reports...")

    for i, idx in enumerate(sampled_indices[:num_samples]):
        row = test_df.loc[idx]
        img_id = row['id_code']
        true_label = CLASS_NAMES[row['diagnosis']]
        img_path = os.path.join("aptos2019-blindness-detection", "train_images", f"{img_id}.png")

        print(f"\n  --- Image {i+1}/{len(sampled_indices)}: {img_id} (True: {true_label}) ---")

        # Load raw BGR image
        bgr_raw = cv2.imread(img_path)
        if bgr_raw is None:
            print(f"  [WARN] Could not load image from '{img_path}'. Skipping.")
            continue

        rgb_raw = cv2.cvtColor(bgr_raw, cv2.COLOR_BGR2RGB)

        # Preprocess image array for neural network inference
        preprocessed_img = preprocessor.preprocess_image(rgb_raw)
        input_tensor = np.expand_dims(preprocessed_img, axis=0)

        # Generate XAI report
        save_file = os.path.join(reports_xai_dir, f"gradcam_{img_id}_true_{row['diagnosis']}.png")
        result = gradcam.generate_xai_report(
            img_array=input_tensor,
            original_img=cv2.resize(rgb_raw, (224, 224)),
            class_names=CLASS_NAMES,
            save_path=save_file
        )

        print(f"  [OK] Predicted: {result['predicted_class']} ({result['confidence']*100:.1f}%)")
        print(f"  [OK] Clinical explanation report saved to: '{save_file}'")

    print(f"\n[OK] Grad-CAM XAI explanation batch complete! All reports stored in '{reports_xai_dir}/'.")


if __name__ == "__main__":
    run_xai_pipeline(num_samples=5)
