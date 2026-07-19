from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from mtcnn import MTCNN
import os
import gdown

IMG_SIZE = 224
THRESHOLD = 0.5

# ✅ GOOGLE DRIVE FILE ID (your link)
FILE_ID = "1WLotK52P5YpeSD-hSjpSHq3KEmA1kBzg"
MODEL_PATH = "autism_model.h5"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

# ================= DOWNLOAD MODEL =================
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("Download complete!")

# ================= FACE DETECTOR =================
detector = None

def get_detector():
    global detector
    if detector is None:
        detector = MTCNN()
    return detector

def extract_face(image):
    try:
        img = np.array(image)
        detector = get_detector()
        results = detector.detect_faces(img)

        if len(results) == 0:
            return image

        x, y, w, h = results[0]['box']
        x, y = abs(x), abs(y)

        face = img[y:y+h, x:x+w]
        return Image.fromarray(face)

    except Exception as e:
        print("Face detection error:", e)
        return image

# ================= MODEL =================
model = None

def load_model():
    global model
    if model is None:
        download_model()

        print("Loading full model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")

    return model

# ================= PREPROCESS =================
from tensorflow.keras.applications.efficientnet import preprocess_input

def preprocess_image(image):
    image = ImageOps.exif_transpose(image)

    face = extract_face(image)
    face = face.convert("RGB")
    face = face.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(face)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# ================= ROUTES =================
@app.route("/")
def home():
    return "API Running Successfully 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]

        image = Image.open(file.stream)
        processed = preprocess_image(image)

        model = load_model()

        prediction = model.predict(processed)[0][0]

        label = "Autistic" if prediction > THRESHOLD else "Non_Autistic"

        return jsonify({
            "prediction": label,
            "confidence": float(prediction)
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)})

# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
