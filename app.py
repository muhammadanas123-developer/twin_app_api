import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import gdown

# 🔥 FORCE CPU (Render fix)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

MODEL_PATH = "autism_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1WLotK52P5YpeSD-hSjpSHq3KEmA1kBzg"

model = None


# ✅ DOWNLOAD MODEL IF NOT EXISTS
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model from Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("✅ Model Downloaded")


# ✅ LOAD MODEL (ONLY ONCE)
def load_my_model():
    global model
    if model is None:
        download_model()
        print("📦 Loading model...")
        model = load_model(MODEL_PATH, compile=False)
        print("✅ Model Loaded Successfully")


# ✅ HOME ROUTE
@app.route("/")
def home():
    return "✅ Autism Prediction API Running"


# ✅ HEALTH CHECK
@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ✅ PREDICTION ROUTE
@app.route("/predict", methods=["POST"])
def predict():
    try:
        load_my_model()

        # ✅ CHECK FILE
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        # ✅ PREPROCESS IMAGE
        img = Image.open(file).convert("RGB")
        img = img.resize((224, 224))

        img = np.array(img).astype("float32")
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        # ✅ PREDICT
        pred = model.predict(img, verbose=0)[0][0]
        pred = float(pred)

        print("🔍 RAW PREDICTION:", pred)

        # 🔥 CORRECT LABEL MAPPING (IMPORTANT)
        # {'Autistic': 0, 'Non_Autistic': 1}
        THRESHOLD = 0.5

        if pred >= THRESHOLD:
            label = "Non-Autistic"
            confidence = pred
        else:
            label = "Autistic"
            confidence = 1 - pred

        # ✅ RESPONSE
        return jsonify({
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "raw": pred
        })

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# ✅ RUN SERVER (Render compatible)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
