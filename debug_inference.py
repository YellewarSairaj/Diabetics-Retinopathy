import os
import hashlib
import numpy as np
import cv2
import pandas as pd
import keras
from training.preprocessing import FundusPreprocessor

def get_checksum(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()

def main():
    print("==================================================")
    print("STARTING END-TO-END INFERENCE DEBUGGING")
    print("==================================================")
    
    # 1. Load trained model
    model_path = os.path.join("models", "final_efficientnet_b0_dr_model.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join("models", "best_efficientnet_b0_stage1.h5")
        
    print(f"[+] Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print(f"[+] Model input shape: {model.input_shape}")
    print(f"[+] Model output shape: {model.output_shape}")
    
    # 2. Initialize preprocessor
    preprocessor = FundusPreprocessor(target_size=(224, 224), apply_crop=True, apply_clahe=True)
    
    # 3. Select 5 sample images from 5 different classes in train.csv
    df = pd.read_csv("aptos2019-blindness-detection/train.csv")
    sample_rows = df.groupby("diagnosis").head(1)
    
    class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    
    print("\n--------------------------------------------------")
    print("TESTING INFERENCE ACROSS 5 DIFFERENT IMAGES")
    print("--------------------------------------------------\n")
    
    for idx, row in sample_rows.iterrows():
        id_code = row["id_code"]
        true_diag = row["diagnosis"]
        img_path = os.path.join("aptos2019-blindness-detection", "train_images", f"{id_code}.png")
        
        if not os.path.exists(img_path):
            continue
            
        with open(img_path, "rb") as f:
            raw_bytes = f.read()
            
        byte_md5 = get_checksum(raw_bytes)
        
        # Read BGR and convert to RGB
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        bgr_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        
        # Raw stats
        raw_mean = np.mean(rgb_img)
        raw_std = np.std(rgb_img)
        raw_min = np.min(rgb_img)
        raw_max = np.max(rgb_img)
        
        # Preprocess
        norm_img = preprocessor.preprocess_image(rgb_img)
        tensor_mean = np.mean(norm_img)
        tensor_std = np.std(norm_img)
        tensor_min = np.min(norm_img)
        tensor_max = np.max(norm_img)
        tensor_md5 = hashlib.md5(norm_img.tobytes()).hexdigest()
        
        input_tensor = np.expand_dims(norm_img, axis=0)
        
        # Run predict
        preds = model.predict(input_tensor, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        pred_class = class_names[pred_idx]
        confidence = float(preds[pred_idx])
        
        print(f"Image ID: {id_code} (True Label: {true_diag} - {class_names[true_diag]})")
        print(f"  Raw File MD5:          {byte_md5}")
        print(f"  Raw Stats:             Shape={rgb_img.shape}, Mean={raw_mean:.2f}, Std={raw_std:.2f}, Min={raw_min}, Max={raw_max}")
        print(f"  Preprocessed Tensor:   Shape={norm_img.shape}, MD5={tensor_md5}")
        print(f"  Preprocessed Stats:     Mean={tensor_mean:.4f}, Std={tensor_std:.4f}, Min={tensor_min:.4f}, Max={tensor_max:.4f}")
        print(f"  Model Predictions:     {np.round(preds, 4).tolist()}")
        print(f"  Predicted Class:       {pred_class} (Conf: {confidence*100:.2f}%)\n")

if __name__ == "__main__":
    main()
