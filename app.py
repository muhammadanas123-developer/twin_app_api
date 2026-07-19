import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

MODEL_PATH = "autism_model.h5"

model = None

# 🔥 FORCE CPU (Render fix)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# ✅ LOAD MODEL (ONLY ONCE)
def load_my_model():
    global model

    if model is None:
        print("📦 Loading model...")
        model = load_model(MODEL_PATH, compile=False)
        print("✅ Model Loaded")


# ✅ HOME
@app.route("/")
def home():
    return "API RUNNING"


# ✅ PREDICT
@app.route("/predict", methods=["POST"])
def predict():
    try:
        load_my_model()

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]

        # 🔥 FIX 1: EXACT SAME PREPROCESS AS TRAIN
        img = Image.open(file).convert("RGB")
        img = img.resize((224, 224))

        img = np.array(img).astype("float32")

        # 🔥 CRITICAL FIX
        img = preprocess_input(img)

        img = np.expand_dims(img, axis=0)

        # 🔥 PREDICT
        pred = float(model.predict(img)[0][0])

        print("RAW PRED:", pred)

        # 🔥 THRESHOLD FIX (IMPORTANT)
        THRESHOLD = 0.35

        if pred < THRESHOLD:
            label = "Autistic"
            confidence = 1 - pred
        else:
            label = "Non-Autistic"
            confidence = pred

        return jsonify({
            "prediction": label,
            "confidence": float(confidence),
            "raw": pred
        })

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
