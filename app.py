import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import gdown

app = Flask(__name__)

MODEL_PATH = "autism_model.h5"
DRIVE_URL = "https://drive.google.com/uc?id=1WLotK52P5YpeSD-hSjpSHq3KEmA1kBzg"

# ✅ Download model if not exists
if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model...")
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
    print("✅ Model downloaded!")

# ✅ Load model only ONCE
print("📦 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded!")

# ✅ Preprocess function (IMPORTANT for matching result)
def preprocess_image(image):
    image = image.resize((224, 224))   # ⚠️ SAME SIZE AS TRAINING
    image = np.array(image) / 255.0    # ⚠️ NORMALIZATION
    image = np.expand_dims(image, axis=0)
    return image

# ✅ Home route
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Autism Detection API",
        "status": "API Running"
    })

# ✅ Predict route (GET + POST)
@app.route("/predict", methods=["GET", "POST"])
def predict():
    # 👉 Browser open kare to ye message aaye
    if request.method == "GET":
        return jsonify({"message": "Use POST with image file"})

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        image = Image.open(file).convert("RGB")
        processed = preprocess_image(image)

        prediction = model.predict(processed)[0][0]
        percentage = float(prediction * 100)

        return jsonify({
            "prediction": "Autistic" if prediction > 0.5 else "Non-Autistic",
            "confidence": round(percentage, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ IMPORTANT for Render (avoid crash)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
