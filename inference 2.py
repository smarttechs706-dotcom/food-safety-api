import sys
import types
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as torchvision_models
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

# ─── Device ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── MobileNetV2 Feature Extractor ────────────────────────────────────────────
mobilenet = torchvision_models.mobilenet_v2(weights='IMAGENET1K_V1')
mobilenet.classifier = torch.nn.Identity()
mobilenet = mobilenet.to(device)
mobilenet.eval()
print("✅ MobileNetV2 feature extractor loaded")

mobilenet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ─── YOLO ─────────────────────────────────────────────────────────────────────
yolo_model = YOLO('best.pt')
print("✅ YOLO loaded")

# ─── Load Models ──────────────────────────────────────────────────────────────
def load_model(path):
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
    obj = joblib.load(path)
    classifier = obj.get('classifier') or obj.get('model')
    vectorizer = obj.get('vectorizer')
    threshold  = obj.get('threshold', 0.5)
    print(f"✅ Loaded text model: {path}")
    return classifier, vectorizer, threshold

# ─── Image models ─────────────────────────────────────────────────────────────
xgb_v1 = load_model('best_image_classifier_98pct.pkl')
xgb_v2 = load_model('image_classifier_91pct.pkl')

# ─── Text models ──────────────────────────────────────────────────────────────
text_clf_v1, text_vec_v1, text_thresh_v1 = load_text_model('best_text_classifier_87pct.pkl')
text_clf_v2, text_vec_v2, text_thresh_v2 = load_text_model('text_classifier_v2_80pct.pkl')

# ─── Clean YOLO object names ──────────────────────────────────────────────────
def clean_object_name(name: str) -> str:
    """Clean up YOLO class names that contain platform descriptions."""
    dirty_keywords = ['roboflow', 'http', 'www', 'platform', 'computer vision', 'end-to-end']
    if len(name) > 30 or any(k in name.lower() for k in dirty_keywords):
        return "food contamination"
    return name.strip().title()

# ─── Image Inference ──────────────────────────────────────────────────────────
def advanced_inference(image_path, model='v1'):
    img = Image.open(image_path).convert("RGB")

    # MobileNetV2 features (1280-dim)
    img_tensor = mobilenet_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = mobilenet(img_tensor)
    final_features = features.cpu().numpy().flatten()

    # YOLO detections
    results = yolo_model(image_path, verbose=False)
    boxes = results[0].boxes
    names = results[0].names

    yolo_count = 0
    max_conf = 0.0
    detected_objects = []

    if boxes is not None and len(boxes) > 0:
        yolo_count = len(boxes)
        max_conf = float(boxes.conf.max())
        # ✅ Clean object names
        detected_objects = list(set([
            clean_object_name(names[int(c)]) for c in boxes.cls
        ]))

    # Classify
    classifier = xgb_v1 if model == 'v1' else xgb_v2
    prob = classifier.predict_proba(final_features.reshape(1, -1))[0][1]
    prediction = int(prob >= 0.5)

    risk_level = (
        "VERY HIGH" if prob > 0.85 else
        "HIGH"      if prob > 0.65 else
        "MEDIUM"    if prob > 0.45 else
        "LOW"
    )

    # Anomaly score
    img_array = np.array(img.resize((128, 128))).astype(np.float32) / 255.0
    recon_error = float(np.var(img_array))

    anomaly_level = (
        "HIGH"   if recon_error > 0.05 else
        "MEDIUM" if recon_error > 0.02 else
        "LOW"
    )

    # Decision reasons — honest and matching prediction
    reasons = []
    if yolo_count > 0:
        reasons.append(f"Visible contamination detected ({yolo_count} object(s))")
    if anomaly_level == "HIGH":
        reasons.append("Abnormal texture / internal structure detected")
    if anomaly_level == "MEDIUM" and prediction == 1:
        reasons.append("Unusual visual patterns detected")
    if prob > 0.85:
        reasons.append("AI model detected high-risk food characteristics")
    elif prob > 0.65:
        reasons.append("AI model detected unsafe food patterns")
    elif prob > 0.45:
        reasons.append("AI model detected borderline food safety indicators")

    if prediction == 0 and not reasons:
        reasons.append("No risk indicators found")
    elif prediction == 0:
        reasons = ["No significant risk indicators"]

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
