from pydantic import BaseModel, Field
from typing import List, Dict, Any
from datetime import datetime


# Pydantic models for API responses
class PredictionResult(BaseModel):
    """Model for prediction response"""
    is_fraud: bool = Field(..., description="Whether the document is predicted to be fraudulent")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    fraud_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of being fraudulent")
    genuine_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of being genuine")
    prediction_class: str = Field(..., description="Predicted class name")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class BatchPredictionResult(BaseModel):
    """Model for batch prediction response"""
    predictions: List[PredictionResult]
    total_processed: int
    processing_time_ms: float
    summary: Dict[str, Any]

class HealthCheck(BaseModel):
    """Model for health check response"""
    status: str
    timestamp: datetime
    model_loaded: bool
    gpu_available: bool
    memory_usage_gb: float
    version: str

class ErrorResponse(BaseModel):
    """Model for error responses"""
    error: str
    detail: str
    timestamp: datetime
    status_code: int