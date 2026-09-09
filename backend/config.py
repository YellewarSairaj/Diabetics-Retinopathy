"""
Backend Service Configuration
==============================
Domain: Healthcare / Computer Vision / Deep Learning
Dataset: APTOS 2019 Blindness Detection
Task: Manage API paths, CORS settings, model configuration, and upload directory settings.

Author: Senior Full-Stack & MLOps Architect
"""

import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Diabetic Retinopathy Explainable AI Platform"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Model Configuration
    MODEL_PATH: str = os.path.join("models", "final_efficientnet_b0_dr_model.keras")
    TARGET_SIZE: tuple = (224, 224)
    NUM_CLASSES: int = 5
    CLASS_NAMES: list = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    
    # Clinical severity descriptions (non-prescriptive decision support)
    SEVERITY_DESCRIPTIONS: dict = {
        "No DR": "No retinal microvascular abnormalities detected on screening.",
        "Mild": "Presence of microaneurysms. Routine ophthalmic evaluation recommended.",
        "Moderate": "Increased microvascular changes observed. Ophthalmologist evaluation recommended.",
        "Severe": "Significant microvascular changes observed. Specialist ophthalmic evaluation recommended.",
        "Proliferative DR": "Highest DR severity grade predicted. Prompt specialist ophthalmic evaluation recommended to determine clinical status."
    }

    # CORS Settings
    ALLOWED_ORIGINS: list = ["*"]
    
    # Temp Upload Storage
    UPLOAD_DIR: str = os.path.join("backend", "temp_uploads")

settings = Settings()
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
