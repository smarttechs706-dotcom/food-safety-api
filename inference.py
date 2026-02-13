import sys
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from ultralytics import YOLO
import joblib
from sklearn.linear_model import LogisticRegression

# Fix slow startup caused by matplotlib and ultralytics config setup
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")

class FoodSafetyClassifier:
    def __init__(self):
        self.model = LogisticRegression()
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def predict(self, X):
        return self.model.predict(X)

sys.modules['__main__'].FoodSafetyClassifier = FoodSafetyClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# ✅ Lazy loading — models are NOT loaded at startup
# They load only when the first request comes in
_ae_model = None
_yolo_model = None
_xgb_v1 = None
_xgb_v2 = None


def get_ae_model():
    """Load autoencoder only when first needed"""
    global _ae_model
    if _ae_model is None:
        _ae_model = Autoencoder().to(device)
        _ae_model.load_state_dict(torch.load('autoencoder.pth', map_location=device))
        _ae_model.eval()
    return _ae_model


def get_yolo_model():
    """Load YOLO only when first needed"""
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO('best.pt')
    return _yolo_model


def get_xgb_v1():
    """Load v1 classifier only when first needed"""
    global _xgb_v1
    if _xgb_v1 is None:
        _xgb_v1 = joblib.load('best_image_classifier_98pct.pkl')
    return _xgb_v1


def get_xgb_v2():
    """Load v2 classifier only when first needed"""
    global _xgb_v2
    if _xgb_v2 is None:
        _xgb_v2 = joblib.load('best_text_classifier_87pct.pkl')
    return _xgb_v2


def advanced_inference(image_path, model='v2'):
    # Load models on first call only
    ae_model = get_ae_model()
    yolo_model = get_yolo_model()

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        encoded = ae_model.encoder(img_tensor)
        reconstructed = ae_model(img_tensor)

    encoded_resized = torch.nn.functional.adaptive_avg_pool2d(encoded, (32, 32))
    latent_features = encoded_resized.flatten().cpu().numpy()

    recon_error = torch.mean((img_tensor - reconstructed) ** 2).item()

    results = yolo_model(image_path, verbose=False)
    boxes = results[0].boxes
    names = results[0].names

    yolo_count = 0
    max_conf = 0.0
    detected_objects = []

    if boxes is not None and len(boxes) > 0:
        yolo_count = len(boxes)
        max_conf = float(boxes.conf.max())
        detected_objects = list(set([names[int(c)] for c in boxes.cls]))

    final_features = np.concatenate([
        latent_features,
        [recon_error, yolo_count, max_conf]
    ])

    if model == 'v1':
        prob = get_xgb_v1().predict_proba(final_features.reshape(1, -1))[0][1]
    else:
        prob = get_xgb_v2().predict_proba(final_features.reshape(1, -1))[0][1]

    prediction = int(prob >= 0.5)

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
        "Status": "UNSAFE ⚠️" if prediction == 1 else "SAFE ✅",
        "Risk Probability": round(float(prob), 2),
        "Risk Level": risk_level,
        "YOLO Detections": yolo_count,
        "Detected Objects": detected_objects,
        "Max Detection Confidence": round(max_conf, 2),
        "Anomaly Score": round(recon_error, 4),
        "Anomaly Level": anomaly_level,
        "Decision Reasons": reasons,
        "Recommended Action": "Reject product" if prediction == 1 else "Approve product",
        "Model Used": model
    }

    return result


if __name__ == "__main__":
    pass
