import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import gdown

app = Flask(__name__)

MODEL_PATH = "autism_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1WLotK52P5YpeSD-hSjpSHq3KEmA1kBzg"

model = None  # ✅ global

def load_my_model():
    global model

    if model is not None:
        return model

    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("✅ Model downloaded!")

    print("📦 Loading model...")
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded!")

    return model


@app.route("/")
def home():
    return "API is running"


@app.route("/predict", methods=["POST"])
def predict():
    print("🔥 PREDICT API HIT")

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        model = load_my_model()  # ✅ load once

        file = request.files['file']

        image = Image.open(file).convert("RGB")
        image = image.resize((224, 224))

        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]

        label = "Autistic" if prediction > 0.5 else "Non-Autistic"
        confidence = float(prediction if prediction > 0.5 else 1 - prediction)

        return jsonify({
            "prediction": label,
            "confidence": confidence
        })

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
