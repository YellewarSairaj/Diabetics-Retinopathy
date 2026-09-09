"""
Dataset Exploration & Pipeline Initialization Module
===================================================
Domain: Healthcare / Computer Vision / Deep Learning
Dataset: APTOS 2019 Blindness Detection (3,662 Retinal Fundus Images)
Task: Load, inspect, summarize, and split raw APTOS 2019 dataset using relative paths.
Author: Senior AI Engineer & Software Architect
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import cv2
from typing import Tuple, Dict

# Standard APTOS 2019 DR Severity Mapping
CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

CLASS_DESCRIPTIONS = {
    0: "No visible signs of diabetic retinopathy.",
    1: "Microaneurysms only (tiny swelling in retinal blood vessels).",
    2: "Microaneurysms, hemorrhages, and hard exudates.",
    3: "Severe intraretinal hemorrhages in 4 quadrants or venous beading.",
    4: "Neovascularization (growth of abnormal fragile blood vessels) or vitreous hemorrhage."
}


class DatasetManager:
    """
    Manages dataset directory loading, image path verification, Exploratory Data
    Analysis (EDA), and Stratified Train/Val/Test splitting for APTOS 2019.
    """

    def __init__(self, dataset_dir: str = "aptos2019-blindness-detection"):
        # Use relative paths as specified in architectural requirements
        self.dataset_dir = dataset_dir
        self.csv_path = os.path.join(self.dataset_dir, "train.csv")
        self.images_dir = os.path.join(self.dataset_dir, "train_images")
        self.ensure_output_directories()

    def ensure_output_directories(self) -> None:
        """Creates necessary project directories for outputs, saved models, and reports."""
        os.makedirs("models", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)
        os.makedirs("reports", exist_ok=True)

    def load_metadata(self) -> pd.DataFrame:
        """
        Loads train.csv metadata from aptos2019-blindness-detection/
        and verifies image file availability on disk.
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"APTOS 2019 metadata file not found at relative path '{self.csv_path}'. "
                "Ensure dataset is extracted in 'aptos2019-blindness-detection/'."
            )

        df = pd.read_csv(self.csv_path)
        print(f"[OK] Successfully loaded train.csv with {len(df)} metadata records.")

        # Inspect and map image file paths dynamically (.png or .jpg)
        df['image_path'] = df['id_code'].apply(self.get_image_file_path)
        
        # Verify image existence
        df['exists'] = df['image_path'].apply(os.path.exists)
        valid_count = df['exists'].sum()
        print(f"[OK] Verified {valid_count}/{len(df)} fundus images exist in '{self.images_dir}'.")

        if valid_count < len(df):
            missing_samples = df[~df['exists']]
            print(f"[WARN] Warning: {len(missing_samples)} missing images detected in dataset folder.")

        return df[df['exists']].copy()

    def get_image_file_path(self, id_code: str) -> str:
        """Dynamically constructs relative image file path without assuming fixed extension."""
        png_path = os.path.join(self.images_dir, f"{id_code}.png")
        jpg_path = os.path.join(self.images_dir, f"{id_code}.jpg")
        if os.path.exists(png_path):
            return png_path
        if os.path.exists(jpg_path):
            return jpg_path
        return png_path  # Default relative fallback

    def explore_dataset(self, df: pd.DataFrame) -> Dict:
        """
        Calculates Exploratory Data Analysis (EDA) metrics: total count,
        per-class distributions, class percentages, and class imbalance ratio.
        """
        print("\n" + "="*60)
        print("    APTOS 2019 DIABETIC RETINOPATHY DATASET EXPLORATION    ")
        print("="*60)

        distribution = df['diagnosis'].value_counts().sort_index()
        total_samples = len(df)

        stats = {
            "total_samples": total_samples,
            "class_counts": {},
            "class_percentages": {}
        }

        print(f"\nTotal Dataset Samples: {total_samples} retinal fundus images\n")
        print(f"{'Class ID':<10} | {'Severity Label':<20} | {'Count':<8} | {'Percentage':<10}")
        print("-" * 56)

        for class_id in range(5):
            count = int(distribution.get(class_id, 0))
            pct = (count / total_samples) * 100 if total_samples > 0 else 0
            label = CLASS_NAMES[class_id]
            stats["class_counts"][label] = count
            stats["class_percentages"][label] = round(pct, 2)
            print(f"{class_id:<10} | {label:<20} | {count:<8} | {pct:.2f}%")

        print("-" * 56)

        majority_count = distribution.max()
        minority_count = distribution.min()
        imbalance_ratio = majority_count / max(minority_count, 1)
        stats["imbalance_ratio"] = round(imbalance_ratio, 2)

        print(f"\nClass Imbalance Ratio (Majority : Minority) -> {imbalance_ratio:.2f}:1")
        print(f"Majority Class: 'No DR' ({majority_count} samples)")
        print(f"Minority Class: 'Severe DR' ({minority_count} samples)")

        if imbalance_ratio > 2.0:
            print("\n[WARN] High class imbalance detected!")
            print("       Strategy: We will compute class weights during training to penalize minority class errors higher.")

        return stats

    def get_stratified_splits(
        self,
        df: pd.DataFrame,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits dataset into Train, Validation, and Test sets using Stratified Sampling
        to preserve exact class distribution ratios across all subsets.
        """
        from sklearn.model_selection import train_test_split

        # Split off test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df['diagnosis'],
            random_state=random_state
        )

        # Split remaining into train and val
        relative_val_size = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_size,
            stratify=train_val_df['diagnosis'],
            random_state=random_state
        )

        print(f"\n[OK] Stratified Dataset Split Completed:")
        print(f"    - Training Set  : {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
        print(f"    - Validation Set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
        print(f"    - Test Set      : {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

        return train_df, val_df, test_df


if __name__ == "__main__":
    manager = DatasetManager(dataset_dir="aptos2019-blindness-detection")
    metadata_df = manager.load_metadata()
    manager.explore_dataset(metadata_df)
    train_df, val_df, test_df = manager.get_stratified_splits(metadata_df)
