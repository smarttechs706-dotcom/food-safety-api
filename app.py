from fastapi import FastAPI, UploadFile, File
import torch
from PIL import Image
import io
import numpy as np
import pickle
import os
import torchvision.transforms as transforms

app = FastAPI(title="Food Safety Prediction API")

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

class Autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 3, 3, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

models = {
    "yolo": None,
    "autoencoder": None,
    "xgb_v1": None,
    "xgb_v2": None,
}

@app.on_event("startup")
def load_models():
    try:
        from ultralytics import YOLO

        if os.path.exists("best.pt"):
            models["yolo"] = YOLO("best.pt")
            print("✅ YOLO loaded")
        else:
            print("⚠️ WARNING: best.pt not found")

        if os.path.exists("autoencoder.pth"):
            models["autoencoder"] = Autoencoder().to(device)
            models["autoencoder"].load_state_dict(
                torch.load("autoencoder.pth", map_location=device)
            )
            models["autoencoder"].eval()
            print("✅ Autoencoder loaded")
        else:
            print("⚠️ WARNING: autoencoder.pth not found")

        if os.path.exists("xgb_model_v1.pkl"):
            with open("xgb_model_v1.pkl", "rb") as f:
                models["xgb_v1"] = pickle.load(f)
            print("✅ XGB v1 loaded")
        else:
            print("⚠️ WARNING: xgb_model_v1.pkl not found")

        if os.path.exists("xgb_model_v2.pkl"):
            with open("xgb_model_v2.pkl", "rb") as f:
                models["xgb_v2"] = pickle.load(f)
            print("✅ XGB v2 loaded")
        else:
            print("⚠️ WARNING: xgb_model_v2.pkl not found")

    except Exception as e:
        print(f"❌ Model loading error: {e}")

@app.get("/")
def root():
    return {"status": "Food Safety API Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Step 1: Autoencoder features
        with torch.no_grad():
            encoded = models["autoencoder"].encoder(img_tensor)
            reconstructed = models["autoencoder"](img_tensor)

        encoded_resized = torch.nn.functional.adaptive_avg_pool2d(encoded, (32, 32))
        latent_features = encoded_resized.flatten().cpu().numpy()

        recon_error = torch.mean((img_tensor - reconstructed) ** 2).item()

        # Step 2: YOLO detection
        yolo_results = models["yolo"](io.BytesIO(image_bytes), verbose=False)
        boxes = yolo_results[0].boxes
        names = yolo_results[0].names

        yolo_count = 0
        max_conf = 0.0
        detected_objects = []

        if boxes is not None and len(boxes) > 0:
            yolo_count = len(boxes)
            max_conf = float(boxes.conf.max())
            detected_objects = list(set([names[int(c)] for c in boxes.cls]))

        # Step 3: Combine features
        final_features = np.concatenate([
            latent_features,
            [recon_error, yolo_count, max_conf]
        ])

        # Step 4: XGBoost prediction (using v2)
        prob = models["xgb_v2"].predict_proba(final_features.reshape(1, -1))[0][1]
        prediction = int(prob >= 0.5)

        # Step 5: Risk levels
        risk_level = (
            "VERY HIGH" if prob > 0.85 else
            "HIGH" if prob > 0.65 else
            "MEDIUM" if prob > 0.45 else
            "LOW"
        )

        anomaly_level = (
            "HIGH" if recon_error > 0.01 else
            "MEDIUM" if recon_error > 0.005 else
            "LOW"
        )

        reasons = []
        if yolo_count > 0:
            reasons.append("Visible contamination detected")
        if anomaly_level == "HIGH":
            reasons.append("Abnormal texture / internal structure")
        if not reasons:
            reasons.append("No risk indicators")

        result = {
            "status": "UNSAFE ⚠️" if prediction == 1 else "SAFE ✅",
            "risk_probability": round(float(prob), 2),
            "risk_level": risk_level,
            "yolo_detections": yolo_count,
            "detected_objects": detected_objects,
            "max_detection_confidence": round(max_conf, 2),
            "anomaly_score": round(recon_error, 4),
            "anomaly_level": anomaly_level,
            "decision_reasons": reasons,
            "recommended_action": "Reject product" if prediction == 1 else "Approve product",
            "prediction": prediction,
            "confidence": round(float(prob), 2)
        }

        return result

    except Exception as e:
        return {"error": str(e)}
