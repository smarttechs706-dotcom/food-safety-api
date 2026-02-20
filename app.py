from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import shutil
import os
from pathlib import Path
import uvicorn
from inference import advanced_inference, text_inference

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Food Safety Detection API",
    description="AI-powered food safety inspection using deep learning models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploaded_images")
UPLOAD_DIR.mkdir(exist_ok=True)

# ─── Response Models ──────────────────────────────────────────────────────────

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

class TextRequest(BaseModel):
    description: str
    model: Optional[str] = 'v1'  # 'v1' = 87%, 'v2' = 80%

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "Food Safety Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/predict":           "POST - Upload image (v1, 98% accuracy)",
            "/predict/v1":        "POST - Upload image using model v1 (98% accuracy)",
            "/predict/v2":        "POST - Upload image using model v2 (91% accuracy)",
            "/predict/text":      "POST - Classify food safety from text (87% accuracy)",
            "/predict/text/v2":   "POST - Classify food safety from text (80% accuracy)",
            "/batch-predict":     "POST - Upload multiple images",
            "/health":            "GET  - Check API health status",
            "/models/info":       "GET  - Get model information",
            "/docs":              "GET  - Interactive API documentation"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": True,
        "message": "All systems operational"
    }

@app.head("/health")
async def health_check_head():
    return Response(status_code=200)

# ─── Image Endpoints ──────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse)
async def predict_default(file: UploadFile = File(...)):
    """Predict food safety using image model v1 (98% accuracy)"""
    return await predict_with_model(file, model='v1')

@app.post("/predict/v1", response_model=PredictionResponse)
async def predict_v1(file: UploadFile = File(...)):
    """Predict food safety using image model v1 (98% accuracy)"""
    return await predict_with_model(file, model='v1')

@app.post("/predict/v2", response_model=PredictionResponse)
async def predict_v2(file: UploadFile = File(...)):
    """Predict food safety using image model v2 (91% accuracy)"""
    return await predict_with_model(file, model='v2')

# ─── Text Endpoints ───────────────────────────────────────────────────────────

@app.post("/predict/text")
async def predict_text_v1(request: TextRequest):
    """Classify food safety from a text description using text model v1 (87% accuracy)"""
    result = text_inference(request.description, model='v1')
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return JSONResponse(content=result)

@app.post("/predict/text/v2")
async def predict_text_v2(request: TextRequest):
    """Classify food safety from a text description using text model v2 (80% accuracy)"""
    result = text_inference(request.description, model='v2')
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return JSONResponse(content=result)

# ─── Helper ───────────────────────────────────────────────────────────────────

async def predict_with_model(file: UploadFile, model: str = 'v1'):
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )

    file_path = UPLOAD_DIR / file.filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = advanced_inference(str(file_path), model=model)
        result['filename'] = file.filename

        formatted_result = {
            "Status":                   result["Status"],
            "Risk_Probability":         result["Risk Probability"],
            "Risk_Level":               result["Risk Level"],
            "YOLO_Detections":          result["YOLO Detections"],
            "Detected_Objects":         result["Detected Objects"],
            "Max_Detection_Confidence": result["Max Detection Confidence"],
            "Anomaly_Score":            result["Anomaly Score"],
            "Anomaly_Level":            result["Anomaly Level"],
            "Decision_Reasons":         result["Decision Reasons"],
            "Recommended_Action":       result["Recommended Action"],
            "Model_Used":               result["Model Used"],
            "filename":                 file.filename
        }

        return JSONResponse(content=formatted_result)

    except Exception as e:
        import traceback
        print(f"ERROR: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

    finally:
        if file_path.exists():
            file_path.unlink()

# ─── Batch ────────────────────────────────────────────────────────────────────

@app.post("/batch-predict")
async def batch_predict(files: List[UploadFile] = File(...), model: str = 'v1'):
    """Predict food safety for multiple images (max 10)"""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")

    results = []
    for file in files:
        try:
            result = await predict_with_model(file, model=model)
            results.append(result)
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e), "Status": "ERROR"})

    return JSONResponse(content={"results": results, "total": len(results)})

# ─── Models Info ──────────────────────────────────────────────────────────────

@app.get("/models/info")
async def models_info():
    return {
        "image_models": {
            "v1": {"name": "best_image_classifier_98pct.pkl", "accuracy": "98%", "type": "Logistic Regression Image Classifier"},
            "v2": {"name": "image_classifier_91pct.pkl",      "accuracy": "91%", "type": "Logistic Regression Image Classifier"}
        },
        "text_models": {
            "v1": {"name": "best_text_classifier_87pct.pkl",  "accuracy": "87%", "type": "Logistic Regression + TF-IDF"},
            "v2": {"name": "text_classifier_v2_80pct.pkl",    "accuracy": "80%", "type": "Logistic Regression + TF-IDF"}
        },
        "supporting_models": {
            "mobilenet":  "MobileNetV2 - Feature extraction (1280 features)",
            "yolo":       "best.pt - Object detection for contamination"
        },
        "pipeline": {
            "image": [
                "1. MobileNetV2 extracts 1280 features from image",
                "2. YOLO detects visible contamination objects",
                "3. LogisticRegression classifies SAFE vs UNSAFE"
            ],
            "text": [
                "1. TF-IDF vectorizer converts text to features",
                "2. LogisticRegression classifies SAFE vs UNSAFE"
            ]
        }
    }

# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
