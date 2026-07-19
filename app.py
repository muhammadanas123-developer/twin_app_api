import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import gdown

app = Flask(__name__)

MODEL_PATH = "autism_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1WLotK52P5YpeSD-hSjpSHq3KEmA1kBzg"

# ✅ Download model if not exists
if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    print("✅ Model downloaded!")

# ✅ Load model
print("📦 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded!")

# ✅ Home route
@app.route("/")
def home():
    return "API is running"

# ✅ Predict route (IMPORTANT FIX)
@app.route("/predict", methods=["GET", "POST"])
def predict():
    print("🔥 PREDICT API HIT")

    if request.method == "GET":
        return jsonify({
            "message": "Use POST with image file"
        })

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
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

# ✅ Run
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
