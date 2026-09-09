"""
Image Preprocessing Pipeline Module
===================================
Domain: Healthcare / Computer Vision / Deep Learning
Dataset: APTOS 2019 Blindness Detection
Task: Implement Ben Graham's fundus preprocessing technique:
      1. Circular bounding-box cropping (remove uninformative black borders)
      2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
      3. Gaussian blur color subtraction for local lesion sharpening
      4. Standardized Resizing (224x224x3)
      5. Min-Max Pixel Normalization [0.0, 1.0] / Model-specific normalization

Author: Senior AI Engineer & Computer Vision Specialist
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class FundusPreprocessor:
    """
    Production-grade ophthalmic image preprocessing engine.
    Applies Ben Graham's cropping, contrast enhancement, resizing, and normalization.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        apply_crop: bool = True,
        apply_clahe: bool = True,
        apply_ben_graham: bool = True,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        sigmaX: float = 30.0
    ):
        self.target_size = target_size
        self.apply_crop = apply_crop
        self.apply_clahe = apply_clahe
        self.apply_ben_graham = apply_ben_graham
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.sigmaX = sigmaX

    def crop_black_borders(self, image: np.ndarray, tol: int = 7) -> np.ndarray:
        """
        Crops uninformative dark background borders around fundus photographs.
        Calculates bounding rectangle based on pixel intensity thresholding.
        """
        if image.ndim == 2:
            mask = image > tol
            return image[np.ix_(mask.any(1), mask.any(0))]
        
        # Grayscale mask to locate non-black eye sphere
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = gray > tol

        check_column = mask.any(1)
        check_row = mask.any(0)

        if not check_column.any() or not check_row.any():
            return image  # Fallback if image is completely dark

        cropped = image[np.ix_(check_column, check_row)]
        return cropped

    def apply_clahe_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
        in the LAB color space (on Luminance channel L) to amplify micro-lesions.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )
        cl = clahe.apply(l_channel)

        enhanced_lab = cv2.merge((cl, a_channel, b_channel))
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        return enhanced_rgb

    def apply_ben_graham_technique(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Ben Graham's Kaggle-winning local color subtraction method:
        I_processed = 4 * I - 4 * GaussianBlur(I, sigmaX) + 128
        Subtracts local background average to bring micro-lesions into high visual contrast.
        """
        blurred = cv2.GaussianBlur(image, (0, 0), self.sigmaX)
        enhanced = cv2.addWeighted(image, 4.0, blurred, -4.0, 128)
        return enhanced

    def preprocess_image(self, image_input) -> np.ndarray:
        """
        Complete end-to-end preprocessing pipeline for a single image.
        Accepts file path (str) or loaded RGB array (np.ndarray).
        Returns float32 normalized image array in shape (H, W, 3).
        """
        # Load image if file path string is passed
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at path: '{image_input}'")
            bgr_image = cv2.imread(image_input)
            if bgr_image is None:
                raise ValueError(f"Failed to decode image file at: '{image_input}'")
            image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        else:
            image = image_input.copy()

        # Step 0: Fast downsampling for high-res images (>512px) to accelerate processing
        h, w = image.shape[:2]
        if max(h, w) > 512:
            scale = 512.0 / float(max(h, w))
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # Step 1: Crop dark background borders
        if self.apply_crop:
            image = self.crop_black_borders(image)

        # Step 2: Apply Ben Graham's local contrast enhancement
        if self.apply_ben_graham:
            image = self.apply_ben_graham_technique(image)
        elif self.apply_clahe:
            image = self.apply_clahe_enhancement(image)

        # Step 3: Resize image to target resolution (224x224) using INTER_AREA
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)

        # Step 4: Convert to float32 range [0.0, 255.0] for Keras EfficientNetB0 internal rescaling layer
        normalized_image = image.astype(np.float32)

        return normalized_image

    def preprocess_batch(self, image_paths: list) -> np.ndarray:
        """Processes a list of image file paths into a 4D batch tensor (B, H, W, C)."""
        processed_list = []
        for path in image_paths:
            img = self.preprocess_image(path)
            processed_list.append(img)
        return np.array(processed_list, dtype=np.float32)


if __name__ == "__main__":
    from dataset import DatasetManager

    print("[+] Initializing Fundus Preprocessing Pipeline Verification...")
    manager = DatasetManager(dataset_dir="aptos2019-blindness-detection")
    df = manager.load_metadata()

    preprocessor = FundusPreprocessor(target_size=(224, 224))
    
    # Test on first image sample
    sample_path = df.iloc[0]['image_path']
    print(f"[+] Testing preprocessor on sample image: '{sample_path}'")

    processed_img = preprocessor.preprocess_image(sample_path)
    
    print("\n[OK] Preprocessing Output Verification:")
    print(f"     - Tensor Shape : {processed_img.shape}")
    print(f"     - Data Type    : {processed_img.dtype}")
    print(f"     - Min Pixel Val: {processed_img.min():.4f}")
    print(f"     - Max Pixel Val: {processed_img.max():.4f}")
    print(f"     - Mean Intensity: {processed_img.mean():.4f}")
