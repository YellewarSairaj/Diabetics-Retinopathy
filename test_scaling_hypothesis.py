import os
import numpy as np
import cv2
import pandas as pd
import keras
from training.preprocessing import FundusPreprocessor

def main():
    print("==================================================")
    print("TESTING SCALING HYPOTHESIS FOR EFFICIENTNET-B0")
    print("==================================================")
    
    model_path = os.path.join("models", "final_efficientnet_b0_dr_model.keras")
    model = keras.models.load_model(model_path)
    
    preprocessor = FundusPreprocessor(target_size=(224, 224), apply_crop=True, apply_clahe=True)
    df = pd.read_csv("aptos2019-blindness-detection/train.csv")
    sample_rows = df.groupby("diagnosis").head(1)
    class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    
    for idx, row in sample_rows.iterrows():
        id_code = row["id_code"]
        true_diag = row["diagnosis"]
        img_path = os.path.join("aptos2019-blindness-detection", "train_images", f"{id_code}.png")
        
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        # 1. Standard pipeline (divided by 255: [0, 1])
        norm_0_1 = preprocessor.preprocess_image(rgb)
        
        # 2. Raw preprocessed (range [0, 255])
        img_0_255 = (norm_0_1 * 255.0).astype(np.float32)
        
        # 3. Keras EfficientNet preprocess_input
        eff_prep = keras.applications.efficientnet.preprocess_input(img_0_255.copy())
        
        p_0_1 = model.predict(np.expand_dims(norm_0_1, 0), verbose=0)[0]
        p_0_255 = model.predict(np.expand_dims(img_0_255, 0), verbose=0)[0]
        p_eff = model.predict(np.expand_dims(eff_prep, 0), verbose=0)[0]
        
        print(f"Image {id_code} (True Label: {true_diag} - {class_names[true_diag]}):")
        print(f"  Range [0, 1] preds   : {np.round(p_0_1, 4).tolist()} -> {class_names[np.argmax(p_0_1)]} ({np.max(p_0_1)*100:.2f}%)")
        print(f"  Range [0, 255] preds : {np.round(p_0_255, 4).tolist()} -> {class_names[np.argmax(p_0_255)]} ({np.max(p_0_255)*100:.2f}%)")
        print(f"  Keras eff_prep preds : {np.round(p_eff, 4).tolist()} -> {class_names[np.argmax(p_eff)]} ({np.max(p_eff)*100:.2f}%)\n")

if __name__ == "__main__":
    main()
