from mtcnn import MTCNN
import numpy as np
from PIL import Image

detector = MTCNN()

def extract_face(image):
    img = np.array(image)
    results = detector.detect_faces(img)

    if len(results) == 0:
        return image  # agar face na mile to original return

    x, y, w, h = results[0]['box']

    x, y = abs(x), abs(y)

    face = img[y:y+h, x:x+w]

    return Image.fromarray(face)
