import os
import hashlib
import numpy as np
import cv2
import pandas as pd
import keras
from training.preprocessing import FundusPreprocessor

def main():
    print("==================================================")
    print("TESTING STAGE 1 MODEL CHECKPOINT")
    print("==================================================")
    
    stage1_path = os.path.join("models", "best_efficientnet_b0_stage1.h5")
    if not os.path.exists(stage1_path):
        print(f"[!] Stage 1 model not found at {stage1_path}")
        return
        
    print(f"[+] Loading Stage 1 model from: {stage1_path}")
    model = keras.models.load_model(stage1_path)
    
    preprocessor = FundusPreprocessor(target_size=(224, 224), apply_crop=True, apply_clahe=True)
    df = pd.read_csv("aptos2019-blindness-detection/train.csv")
    sample_rows = df.groupby("diagnosis").head(1)
    class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    
    for idx, row in sample_rows.iterrows():
        id_code = row["id_code"]
        true_diag = row["diagnosis"]
        img_path = os.path.join("aptos2019-blindness-detection", "train_images", f"{id_code}.png")
        
        with open(img_path, "rb") as f:
            raw_bytes = f.read()
            
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        bgr_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        
        norm_img = preprocessor.preprocess_image(rgb_img)
        input_tensor = np.expand_dims(norm_img, axis=0)
        
        preds = model.predict(input_tensor, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        pred_class = class_names[pred_idx]
        confidence = float(preds[pred_idx])
        
        print(f"Image ID: {id_code} (True Label: {true_diag} - {class_names[true_diag]})")
        print(f"  Model Predictions:     {np.round(preds, 4).tolist()}")
        print(f"  Predicted Class:       {pred_class} (Conf: {confidence*100:.2f}%)\n")

if __name__ == "__main__":
    main()
