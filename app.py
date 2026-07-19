from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os
import gdown

# ================= CONFIG =================
IMG_SIZE = 224
THRESHOLD = 0.5

FILE_ID = "1WLotK52P5YpeSD-hSjpSHq3KEmA1kBzg"
MODEL_PATH = "autism_model.h5"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = Flask(__name__)

# ================= MODEL =================
model = None

def load_model():
    global model

    if model is None:

        # Download if not exists
        if not os.path.exists(MODEL_PATH):
            print("📥 Downloading model...")
            url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
            print("✅ Download complete!")

        print("📦 Loading model...")
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )
        print("✅ Model loaded!")

    return model

# ================= PREPROCESS =================
from tensorflow.keras.applications.efficientnet import preprocess_input

def preprocess_image(image):

    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(image)

    if img_array is None or img_array.size == 0:
        raise ValueError("Invalid image")

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

# 🔥 TEMP DEBUG (GET + POST both)
@app.route("/predict", methods=["GET", "POST"])
def predict():
    try:

        if request.method == "GET":
            return jsonify({"message": "Use POST with image file"})

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        image = Image.open(file.stream)
        processed = preprocess_image(image)

        model = load_model()

        prediction = model.predict(processed)[0][0]

        confidence = float(prediction)

        label = "Autistic" if confidence > THRESHOLD else "Non_Autistic"

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
