"""
Pydantic API Request/Response Schemas
=====================================
Domain: Healthcare / Computer Vision / Deep Learning
Dataset: APTOS 2019 Blindness Detection
Task: Define strongly typed data validation contracts for API requests/responses.

Author: Senior Software Architect & Full-Stack Engineer
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class ClassProbability(BaseModel):
    class_name: str = Field(..., example="Moderate")
    class_index: int = Field(..., example=2)
    probability: float = Field(..., example=0.8742)

class HealthCheckResponse(BaseModel):
    status: str = Field(..., example="healthy")
    project_name: str
    version: str
    model_loaded: bool
    model_architecture: str

class PredictionResponse(BaseModel):
    filename: str = Field(..., example="fundus_sample_01.png")
    predicted_class: str = Field(..., example="Moderate")
    predicted_index: int = Field(..., example=2)
    confidence: float = Field(..., example=0.8742)
    clinical_recommendation: str = Field(..., example="Early clinical intervention recommended.")
    class_probabilities: List[ClassProbability]

class ExplainResponse(BaseModel):
    prediction: PredictionResponse
    gradcam_heatmap_base64: str = Field(..., description="Base64 encoded PNG of thermal Grad-CAM heatmap")
    clinical_overlay_base64: str = Field(..., description="Base64 encoded PNG of fundus image with Grad-CAM overlay")
    xai_layer_used: str = Field(..., example="top_activation")
