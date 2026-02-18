import sys
import types
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from ultralytics import YOLO
import joblib
from sklearn.linear_model import LogisticRegression

# ─── Keras Stub ───────────────────────────────────────────────────────────────
def _make_keras_stub():
    keras = types.ModuleType("keras")
    for sub in [
        "keras.engine", "keras.engine.sequential", "keras.engine.training",
        "keras.layers", "keras.layers.core", "keras.layers.convolutional",
        "keras.layers.pooling", "keras.layers.recurrent",
        "keras.layers.normalization", "keras.layers.merge",
        "keras.optimizers", "keras.models", "keras.utils",
        "keras.utils.generic_utils", "keras.src", "keras.src.engine",
        "keras.src.layers", "keras.src.models", "keras.src.optimizers",
        "keras.src.utils",
    ]:
        mod = types.ModuleType(sub)
        sys.modules[sub] = mod
    sys.modules["keras"] = keras

_make_keras_stub()

# ─── FoodSafetyClassifier stub ────────────────────────────────────────────────
class FoodSafetyClassifier:
    def __init__(self):
        self.model = LogisticRegression()

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)

sys.modules['__main__'].FoodSafetyClassifier = FoodSafetyClassifier

# ─── Device & transforms ──────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ─── Autoencoder ──────────────────────────────────────────────────────────────
class Autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, 3, padding=1), torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 3, 3, padding=1), torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ─── Load models ──────────────────────────────────────────────────────────────
ae_model = Autoencoder().to(device)
ae_model.load_state_dict(torch.load('autoencoder.pth', map_location=device))
ae_model.eval()
print("✅ Autoencoder loaded")

yolo_model = YOLO('best.pt')
print("✅ YOLO loaded")

def load_model(path):
    """Safely load a model — handles plain models, dict wrappers, and pipelines."""
    obj = joblib.load(path)
    if isinstance(obj, dict):
        for key in ['model', 'classifier', 'clf', 'estimator']:
            if key in obj and hasattr(obj[key], 'predict_proba'):
                print(f"✅ Loaded model from dict key '{key}': {path}")
                return obj[key]
        for key, val in obj.items():
            if hasattr(val, 'predict_proba'):
                print(f"✅ Loaded model from dict key '{key}': {path}")
                return val
        raise ValueError(f"❌ No valid model found in dict from {path}. Keys: {list(obj.keys())}")
    print(f"✅ Loaded model: {path}")
    return obj

def load_text_model(path):
    """Load a text model bundle (classifier + vectorizer)."""
    obj = joblib.load(path)
    classifier = obj.get('classifier') or obj.get('model')
    vectorizer = obj.get('vectorizer')
    threshold  = obj.get('threshold', 0.5)
    print(f"✅ Loaded text model: {path}")
    return classifier, vectorizer, threshold

# ─── Image models ─────────────────────────────────────────────────────────────
xgb_v1 = load_model('best_image_classifier_98pct.pkl')   # 98% — primary
xgb_v2 = load_model('image_classifier_91pct.pkl')        # 91% — secondary

# ─── Text models ──────────────────────────────────────────────────────────────
text_clf_v1, text_vec_v1, text_thresh_v1 = load_text_model('best_text_classifier_87pct.pkl')   # 87%
text_clf_v2, text_vec_v2, text_thresh_v2 = load_text_model('text_classifier_v2_80pct.pkl')     # 80%

# ─── Image Inference ──────────────────────────────────────────────────────────
def advanced_inference(image_path, model='v1'):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        encoded = ae_model.encoder(img_tensor)
        reconstructed = ae_model(img_tensor)

    encoded_resized = torch.nn.functional.adaptive_avg_pool2d(encoded, (4, 5))
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

    # ✅ Use ONLY the latent features (1280 features total, no extras)
    final_features = latent_features

    classifier = xgb_v1 if model == 'v1' else xgb_v2
    prob = classifier.predict_proba(final_features.reshape(1, -1))[0][1]
    prediction = int(prob >= 0.5)

    risk_level = (
        "VERY HIGH" if prob > 0.85 else
        "HIGH"      if prob > 0.65 else
        "MEDIUM"    if prob > 0.45 else
        "LOW"
    )

    anomaly_level = (
        "HIGH"   if recon_error > 0.01  else
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

    return {
        "Status":                   "UNSAFE ⚠️" if prediction == 1 else "SAFE ✅",
        "Risk Probability":         round(float(prob), 2),
        "Risk Level":               risk_level,
        "YOLO Detections":          yolo_count,
        "Detected Objects":         detected_objects,
        "Max Detection Confidence": round(max_conf, 2),
        "Anomaly Score":            round(recon_error, 4),
        "Anomaly Level":            anomaly_level,
        "Decision Reasons":         reasons,
        "Recommended Action":       "Reject product" if prediction == 1 else "Approve product",
        "Model Used":               f"image_{model}"
    }

# ─── Text Inference ───────────────────────────────────────────────────────────
def text_inference(description: str, model='v1'):
    """
    Classify food safety from a text description.
    model='v1' uses best_text_classifier_87pct (87% accuracy)
    model='v2' uses text_classifier_v2_80pct   (80% accuracy)
    """
    if not description or not description.strip():
        return {"error": "Description cannot be empty"}

    if model == 'v1':
        classifier, vectorizer, threshold = text_clf_v1, text_vec_v1, text_thresh_v1
    else:
        classifier, vectorizer, threshold = text_clf_v2, text_vec_v2, text_thresh_v2

    X = vectorizer.transform([description])
    prob = classifier.predict_proba(X)[0][1]
    prediction = int(prob >= threshold)

    risk_level = (
        "VERY HIGH" if prob > 0.85 else
        "HIGH"      if prob > 0.65 else
        "MEDIUM"    if prob > 0.45 else
        "LOW"
    )

    return {
        "Status":              "UNSAFE ⚠️" if prediction == 1 else "SAFE ✅",
        "Risk Probability":    round(float(prob), 2),
        "Risk Level":          risk_level,
        "Recommended Action":  "Reject product" if prediction == 1 else "Approve product",
        "Model Used":          f"text_{model}",
        "Input Description":   description
    }


if __name__ == "__main__":
    pass
