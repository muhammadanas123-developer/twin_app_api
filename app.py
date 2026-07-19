import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import gdown

app = Flask(__name__)

MODEL_PATH = "autism_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1WLotK52P5YpeSD-hSjpSHq3KEmA1kBzg"

model = None


# ✅ LOAD MODEL
def load_my_model():
    global model

    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("✅ Model downloaded!")

    if model is None:
        print("📦 Loading model...")
        
        # 🔥 FIX HERE
        model = load_model(MODEL_PATH, compile=False)
        
        print("✅ Model loaded!")


# ✅ HOME
@app.route("/")
def home():
    return "API is running"


# ✅ PREDICT
@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("🔥 PREDICT API HIT")

        load_my_model()

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]

        # ✅ IMAGE PROCESSING (IMPORTANT FIX)
        img = Image.open(file).convert("RGB")
        img = img.resize((224, 224))

        img_array = np.array(img)
        img_array = preprocess_input(img_array)   # 🔥 IMPORTANT
        img_array = np.expand_dims(img_array, axis=0)

        # ✅ PREDICT
        pred = model.predict(img_array)[0][0]

        print("🧠 RAW PRED:", pred)

        # ✅ LABEL FIX
        if pred >= 0.5:
            label = "Autistic"
            confidence = float(pred)
        else:
            label = "Non-Autistic"
            confidence = float(1 - pred)

        return jsonify({
            "prediction": label,
            "confidence": confidence
        })

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
