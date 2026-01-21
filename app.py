from fastapi import FastAPI, UploadFile, File
import torch
from PIL import Image
import io
import numpy as np
import pickle
import os

# FastAPI app
app = FastAPI(title="Food Safety Prediction API")

# Models container
models = {
    "yolo": None,
    "autoencoder": None,
    "xgb_v1": None,
    "xgb_v2": None,
}

# -----------------------------
# Startup: load models safely
# -----------------------------
@app.on_event("startup")
def load_models():
    try:
        from ultralytics import YOLO

        # YOLO
        if os.path.exists("best.pt"):
            models["yolo"] = YOLO("best.pt")
            print("YOLO loaded")
        else:
            print("WARNING: best.pt not found")

        # Autoencoder
        if os.path.exists("autoencoder.pth"):
            models["autoencoder"] = torch.load(
                "autoencoder.pth", map_location="cpu", weights_only=False
            )
            models["autoencoder"].eval()
            print("Autoencoder loaded")
        else:
            print("WARNING: autoencoder.pth not found")

        # XGBoost v1
        if os.path.exists("xgb_model_v1.pkl"):
            with open("xgb_model_v1.pkl", "rb") as f:
                models["xgb_v1"] = pickle.load(f)
            print("XGB v1 loaded")
        else:
            print("WARNING: xgb_model_v1.pkl not found")

        # XGBoost v2
        if os.path.exists("xgb_model_v2.pkl"):
            with open("xgb_model_v2.pkl", "rb") as f:
                models["xgb_v2"] = pickle.load(f)
            print("XGB v2 loaded")
        else:
            print("WARNING: xgb_model_v2.pkl not found")

    except Exception as e:
        print("Model loading error:", e)

# -----------------------------
# Root endpoint
# -----------------------------
@app.get("/")
def root():
    return {"status": "server running"}

# -----------------------------
# Helper: preprocess image
# -----------------------------
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    array = np.array(image).astype(np.float32) / 255.0
    tensor = torch.tensor(array).permute(2, 0, 1).unsqueeze(0)
    return tensor

# -----------------------------
# Predict endpoint
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    response = {"filename": file.filename}

    # Read image bytes
    image_bytes = await file.read()
    image_tensor = preprocess_image(image_bytes)

    # -----------------------------
    # YOLO detection
    # -----------------------------
    yolo_results_list = []
    if models["yolo"]:
        try:
            yolo_results = models["yolo"].predict(io.BytesIO(image_bytes))
            for det in yolo_results:
                boxes = det.boxes
                if boxes is not None and len(boxes) > 0:
                    for i in range(len(boxes)):
                        cls = int(boxes.cls[i])
                        conf = float(boxes.conf[i])
                        label = det.names.get(cls, f"class_{cls}")
                        yolo_results_list.append({
                            "label": label,
                            "confidence": conf
                        })
        except Exception as e:
            yolo_results_list = {"error": str(e)}
    else:
        yolo_results_list = "YOLO model not loaded"

    response["yolo"] = yolo_results_list

    # -----------------------------
    # Autoencoder features
    # -----------------------------
    autoencoder_features = None
    if models["autoencoder"]:
        try:
            with torch.no_grad():
                features = models["autoencoder"](image_tensor)
                autoencoder_features = features.flatten().cpu().numpy().tolist()
        except Exception as e:
            autoencoder_features = {"error": str(e)}

    response["autoencoder_features"] = autoencoder_features

    # -----------------------------
    # XGBoost predictions
    # -----------------------------
    xgb_predictions = {}
    for key in ["xgb_v1", "xgb_v2"]:
        model = models[key]
        if model:
            try:
                flat_input = image_tensor.flatten().numpy().reshape(1, -1)
                xgb_predictions[key] = model.predict(flat_input).tolist()
            except Exception as e:
                xgb_predictions[key] = {"error": str(e)}
        else:
            xgb_predictions[key] = None

    response["xgb_predictions"] = xgb_predictions

    return response
