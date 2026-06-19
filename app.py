from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os

app = Flask(__name__)
CORS(app)

# load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model_hama_pakcoy.keras")

model = tf.keras.models.load_model(model_path)

classes = ["siput", "tanpa_hama", "ulat_grayak"]


# fungsi cek apakah gambar daun pakcoy
def is_pakcoy_leaf(img_array):
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    green_ratio = np.sum(mask > 0) / mask.size

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return False

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    image_area = img_array.shape[0] * img_array.shape[1]
    area_ratio = area / image_area

    if green_ratio > 0.25 and area_ratio > 0.15:
        return True

    return False


@app.route("/")
def home():
    return "Flask AI Pakcoy Shield berjalan!"


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image") or request.files.get("file")

    if file is None:
        return jsonify({"error": "File gambar tidak ditemukan. Gunakan key 'image' atau 'file'"}), 400

    img = Image.open(file).convert("RGB")
    img = img.resize((299, 299))

    img_array = np.array(img)

    # cek apakah daun pakcoy
    if not is_pakcoy_leaf(img_array):
        return jsonify({
            "prediction": "Gambar tidak bisa diproses",
            "confidence": 0
        })

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    confidence = float(np.max(prediction)) * 100
    index = np.argmax(prediction)

    hasil = classes[index]

    if hasil != "tanpa_hama" and confidence < 30:
        hasil = "Hama tidak terdeteksi"

    return jsonify({
        "prediction": hasil,
        "confidence": round(confidence, 2)
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)