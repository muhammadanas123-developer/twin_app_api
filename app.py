from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os
import gdown

# ================= CONFIG =================
IMG_SIZE = 224
THRESHOLD = 0.5
MODEL_PATH = "autism_model.h5"

# ✅ YOUR GOOGLE DRIVE FILE ID
FILE_ID = "1WLotK52P5YpeSD-hSjpSHq3KEmA1kBzg"

app = Flask(__name__)

# ================= DOWNLOAD MODEL =================
if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
    print("✅ Model downloaded!")

# ================= LOAD MODEL =================
print("📦 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded!")

# ✅ SAME AS TRAINING
from tensorflow.keras.applications.efficientnet import preprocess_input

# ================= PREPROCESS =================
def preprocess_image(image):
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(image)

    # 🔥 IMPORTANT (same as training)
    img_array = preprocess_input(img_array)

    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# ================= ROUTES =================

@app.route("/")
def home():
    return jsonify({
        "status": "API Running",
        "message": "Autism Detection API"
    })

# ✅ FIXED ROUTE (GET + POST)
@app.route("/predict", methods=["GET", "POST"])
def predict():

    # 👉 Browser access fix (no more 405 error)
    if request.method == "GET":
        return jsonify({"message": "Use POST with image file"})

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        image = Image.open(file.stream)

        processed = preprocess_image(image)

        prediction = model.predict(processed)

        confidence = float(prediction[0][0])

        label = "Autistic" if confidence > THRESHOLD else "Non_Autistic"

        print("✅ Prediction:", label, "| Confidence:", confidence)

        return jsonify({
            "prediction": label,
            "confidence": confidence
        })

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({"error": str(e)}), 500

# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
