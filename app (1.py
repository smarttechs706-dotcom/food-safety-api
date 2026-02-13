from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import shutil
import os
from pathlib import Path
import uvicorn
from inference import advanced_inference

# Create FastAPI app
app = FastAPI(
    title="Food Safety Detection API",
    description="AI-powered food safety inspection using deep learning models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploaded_images")
UPLOAD_DIR.mkdir(exist_ok=True)

# Response model
class PredictionResponse(BaseModel):
    Status: str
    Risk_Probability: float
    Risk_Level: str
    YOLO_Detections: int
    Detected_Objects: List[str]
    Max_Detection_Confidence: float
    Anomaly_Score: float
    Anomaly_Level: str
    Decision_Reasons: List[str]
    Recommended_Action: str
    Model_Used: str
    filename: str

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Food Safety Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Upload image for safety prediction",
            "/predict/v1": "POST - Upload image using model v1 (98% accuracy)",
            "/predict/v2": "POST - Upload image using model v2 (87% accuracy)",
            "/health": "GET - Check API health status",
            "/docs": "GET - Interactive API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": True,
        "message": "All systems operational"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_default(file: UploadFile = File(...)):
    """
    Predict food safety using default model (v2)
    
    - **file**: Image file to analyze (jpg, jpeg, png)
    """
    return await predict_with_model(file, model='v2')

@app.post("/predict/v1", response_model=PredictionResponse)
async def predict_v1(file: UploadFile = File(...)):
    """
    Predict food safety using model v1 (best_image_classifier_98pct)
    
    - **file**: Image file to analyze (jpg, jpeg, png)
    """
    return await predict_with_model(file, model='v1')

@app.post("/predict/v2", response_model=PredictionResponse)
async def predict_v2(file: UploadFile = File(...)):
    """
    Predict food safety using model v2 (best_text_classifier_87pct)
    
    - **file**: Image file to analyze (jpg, jpeg, png)
    """
    return await predict_with_model(file, model='v2')

async def predict_with_model(file: UploadFile, model: str = 'v2'):
    """
    Helper function to process image and run inference
    """
    # Validate file type
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file temporarily
    file_path = UPLOAD_DIR / file.filename
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run inference
        result = advanced_inference(str(file_path), model=model)
        
        # Add filename to result
        result['filename'] = file.filename
        
        # Clean up field names for response model (replace spaces with underscores)
        formatted_result = {
            "Status": result["Status"],
            "Risk_Probability": result["Risk Probability"],
            "Risk_Level": result["Risk Level"],
            "YOLO_Detections": result["YOLO Detections"],
            "Detected_Objects": result["Detected Objects"],
            "Max_Detection_Confidence": result["Max Detection Confidence"],
            "Anomaly_Score": result["Anomaly Score"],
            "Anomaly_Level": result["Anomaly Level"],
            "Decision_Reasons": result["Decision Reasons"],
            "Recommended_Action": result["Recommended Action"],
            "Model_Used": result["Model Used"],
            "filename": file.filename
        }
        
        return JSONResponse(content=formatted_result)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )
    
    finally:
        # Clean up uploaded file
        if file_path.exists():
            file_path.unlink()

@app.post("/batch-predict")
async def batch_predict(files: List[UploadFile] = File(...), model: str = 'v2'):
    """
    Predict food safety for multiple images
    
    - **files**: List of image files to analyze
    - **model**: Model version to use ('v1' or 'v2')
    """
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 files allowed per batch"
        )
    
    results = []
    
    for file in files:
        try:
            result = await predict_with_model(file, model=model)
            results.append(result)
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "Status": "ERROR"
            })
    
    return JSONResponse(content={"results": results, "total": len(results)})

@app.get("/models/info")
async def models_info():
    """Get information about loaded models"""
    return {
        "models": {
            "v1": {
                "name": "best_image_classifier_98pct.pkl",
                "accuracy": "98%",
                "type": "XGBoost Image Classifier"
            },
            "v2": {
                "name": "best_text_classifier_87pct.pkl",
                "accuracy": "87%",
                "type": "XGBoost Text Classifier"
            }
        },
        "supporting_models": {
            "autoencoder": "autoencoder.pth - Feature extraction and anomaly detection",
            "yolo": "best.pt - Object detection for contamination"
        },
        "pipeline": [
            "1. Autoencoder extracts latent features and computes reconstruction error",
            "2. YOLO detects visible contamination objects",
            "3. XGBoost classifier combines features for final prediction"
        ]
    }

if __name__ == "__main__":
    # Run the application - FIXED: Use dynamic PORT from Render
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
