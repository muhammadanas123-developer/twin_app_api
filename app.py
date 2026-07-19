import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import gdown

app = Flask(__name__)

MODEL_PATH = "autism_model.h5"
DRIVE_URL = "https://drive.google.com/uc?id=1WLotK52P5YpeSD-hSjpSHq3KEmA1kBzg"

# Download model
if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model...")
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
    print("✅ Model downloaded!")

# Load model
print("📦 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded!")

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API Running"})

@app.route("/predict", methods=["GET", "POST"])
def predict():

    print("🔥 PREDICT API HIT")

    if request.method == "GET":
        return jsonify({"message": "Use POST with image file"})

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        image = Image.open(file).convert("RGB")

        processed = preprocess_image(image)

        prediction = model.predict(processed)[0][0]
        confidence = float(prediction * 100)

        print(f"Prediction: {prediction}, Confidence: {confidence}")

        return jsonify({
            "prediction": "Autistic" if prediction > 0.5 else "Non-Autistic",
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
