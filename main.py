from fastapi import FastAPI, File, UploadFile, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import logging
import asyncio
import time
from datetime import datetime, timezone
import gc
import psutil
import os
from pathlib import Path
import uvicorn
from validation_schema import (
    PredictionResult,
    BatchPredictionResult,
    HealthCheck,
    ErrorResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="Document Fraud Detection API",
    description="Federated Learning-based Document Fraud Detection Service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model Architecture
class ImprovedCNN(nn.Module):
    """Improved CNN architecture for document fraud detection"""
    
    def __init__(self, num_classes=2):
        super(ImprovedCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



# Global variables for model and device
MODEL = None
DEVICE = None
TRANSFORM = None

# Configuration
MODEL_PATH = "federated_document_fraud_detector.pth"
MAX_FILE_SIZE = 10 * 1024 * 1024  
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
CLASS_NAMES = {0: "genuine", 1: "fraud"}

def setup_device() -> torch.device:
    """Setup computation device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    return device

def get_data_transforms():
    """Get data preprocessing transforms"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform

def load_model() -> bool:
    """Load the trained model"""
    global MODEL, DEVICE, TRANSFORM
    
    try:
        # Setup device and transforms
        DEVICE = setup_device()
        TRANSFORM = get_data_transforms()
        
        # Initialize model
        MODEL = ImprovedCNN(num_classes=2)
        
        # Load model weights if file exists
        model_path = Path(MODEL_PATH)
        if model_path.exists():
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            MODEL.load_state_dict(state_dict)
            logger.info(f"Model loaded from {MODEL_PATH}")
        else:
            logger.warning(f"Model file {MODEL_PATH} not found. Using untrained model.")
        
        MODEL.to(DEVICE)
        MODEL.eval()
        
        logger.info("Model setup completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

def get_memory_usage() -> float:
    """Get current memory usage in GB"""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024 / 1024
    except:
        return 0.0

def validate_image_file(file: UploadFile) -> bool:
    """Validate uploaded image file"""
    # Check file extension
    file_ext = Path(file.filename or "").suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return False
    
    # Check file size
    if hasattr(file, 'size') and file.size > MAX_FILE_SIZE:
        return False
    
    return True

async def preprocess_image(file_content: bytes) -> torch.Tensor:
    """Preprocess image for model inference"""
    try:
        # Load image
        image = Image.open(io.BytesIO(file_content)).convert('RGB')
        
        # Apply transforms
        if TRANSFORM:
            image_tensor = TRANSFORM(image).unsqueeze(0)  # Add batch dimension
            return image_tensor.to(DEVICE)
        else:
            raise ValueError("Transform not initialized")
            
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image format: {str(e)}"
        )

async def predict_single_image(image_tensor: torch.Tensor) -> Dict[str, Any]:
    """Make prediction on single image"""
    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    start_time = time.time()
    
    try:
        with torch.no_grad():
            # Get model output
            outputs = MODEL(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            predicted_class = predicted.item()
            
            # Get probabilities
            probs = probabilities.cpu().numpy()[0]
            genuine_prob = float(probs[0])
            fraud_prob = float(probs[1])
            
            # Determine result
            is_fraud = predicted_class == 1
            confidence = max(genuine_prob, fraud_prob)
            prediction_class = CLASS_NAMES[predicted_class]
            
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return {
                "is_fraud": is_fraud,
                "confidence": confidence,
                "fraud_probability": fraud_prob,
                "genuine_probability": genuine_prob,
                "prediction_class": prediction_class,
                "processing_time_ms": processing_time
            }
            
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )
    finally:
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting Document Fraud Detection API...")
    
    success = load_model()
    if not success:
        logger.error("Failed to load model during startup")
    else:
        logger.info("API startup completed successfully")

@app.get("/health", response_model=HealthCheck)
async def health_check() -> HealthCheck:
    """Health check endpoint"""
    return HealthCheck(
        status="healthy" if MODEL is not None else "unhealthy",
        timestamp=datetime.now(timezone.utc),
        model_loaded=MODEL is not None,
        gpu_available=torch.cuda.is_available(),
        memory_usage_gb=get_memory_usage(),
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResult)
async def predict_document(file: UploadFile = File(...)) -> PredictionResult:
    """
    Predict if a document image is fraudulent or genuine
    
    - **file**: Document image file (JPG, PNG, BMP, TIFF)
    - Returns prediction result with confidence scores
    """
    # Validate file
    if not validate_image_file(file):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format or size. Supported: JPG, PNG, BMP, TIFF (max 10MB)"
        )
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Preprocess image
        image_tensor = await preprocess_image(file_content)
        
        # Make prediction
        result = await predict_single_image(image_tensor)
        
        return PredictionResult(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during prediction"
        )

@app.post("/predict/batch", response_model=BatchPredictionResult)
async def predict_batch_documents(files: List[UploadFile] = File(...)) -> BatchPredictionResult:
    """
    Predict multiple document images in batch
    
    - **files**: List of document image files
    - Returns batch prediction results with summary
    """
    if len(files) > 10:  
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size limited to 10 files"
        )
    
    start_time = time.time()
    predictions = []
    fraud_count = 0
    genuine_count = 0
    
    for i, file in enumerate(files):
        try:
            # Validate file
            if not validate_image_file(file):
                logger.warning(f"Skipping invalid file {i}: {file.filename}")
                continue
            
            # Read and process file
            file_content = await file.read()
            image_tensor = await preprocess_image(file_content)
            
            # Make prediction
            result = await predict_single_image(image_tensor)
            predictions.append(PredictionResult(**result))
            
            # Update counters
            if result["is_fraud"]:
                fraud_count += 1
            else:
                genuine_count += 1
                
        except Exception as e:
            logger.error(f"Error processing file {i}: {e}")
            continue
    
    total_time = (time.time() - start_time) * 1000
    
    # Create summary
    summary = {
        "total_files": len(files),
        "processed_files": len(predictions),
        "fraud_detected": fraud_count,
        "genuine_documents": genuine_count,
        "fraud_rate": fraud_count / len(predictions) if predictions else 0,
        "average_confidence": sum(p.confidence for p in predictions) / len(predictions) if predictions else 0
    }
    
    return BatchPredictionResult(
        predictions=predictions,
        total_processed=len(predictions),
        processing_time_ms=total_time,
        summary=summary
    )

@app.get("/model/info")
async def get_model_info() -> Dict[str, Any]:
    """Get model information and statistics"""
    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in MODEL.parameters())
    trainable_params = sum(p.numel() for p in MODEL.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / 1024 / 1024  # Assuming float32
    
    return {
        "model_name": "ImprovedCNN",
        "model_type": "Federated Learning Document Fraud Detector",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": round(model_size_mb, 2),
        "input_size": [3, 224, 224],
        "num_classes": 2,
        "class_names": CLASS_NAMES,
        "device": str(DEVICE),
        "loaded_from": MODEL_PATH if Path(MODEL_PATH).exists() else "default_weights"
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc.detail),
            timestamp=datetime.now(timezone.utc),
            status_code=exc.status_code
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred",
            timestamp=datetime.now(timezone.utc),
            status_code=500
        ).dict()
    )

if __name__ == "__main__":
    # Production server configuration
    uvicorn.run("main:app",host="0.0.0.0",port=8000,reload=False,workers=1,log_level="info",access_log=True)