from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from mtcnn import MTCNN
import os
import gdown

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = 224
THRESHOLD = 0.5

# ✅ YOUR FILE ID
FILE_ID = "1aN7_TYwhbL0GUYmiD591T1BE2bDNyfjD"
MODEL_PATH = "best_weights.weights.h5"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

# ================= DOWNLOAD MODEL =================
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
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

        base_model = EfficientNetB0(
            weights=None,
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)

        output = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=base_model.input, outputs=output)

        print("Loading weights...")
        model.load_weights(MODEL_PATH)
        print("Model loaded!")

    return model

# ================= PREPROCESS =================
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
