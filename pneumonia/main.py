import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import numpy as np
from PIL import Image

from flask import Flask, request, jsonify
from flask_cors import CORS

unique_labels = ["NORMAL", "PNEUMONIA"]

IMG_SIZE = 224


def process_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    return image


BATCH_SIZE = 32


def create_data_batches(X, batch_size=BATCH_SIZE):
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))
    data_batch = data.map(process_image).batch(BATCH_SIZE)
    return data_batch


def get_pred_label(prediction_probabilities):
    if prediction_probabilities[np.argmax(prediction_probabilities)] >= 0.6:
        return unique_labels[np.argmax(prediction_probabilities)]
    else:
        return "no_tumor"


def load_model(model_path):
    print(f"Loading Saved Model From: {model_path}")
    model = tf.keras.models.load_model(
        model_path, custom_objects={"KerasLayer": hub.KerasLayer}
    )
    return model


model = load_model("model.h5")

# ************************************************************ #

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})
        try:
            image_bytes = file.read()
            data = create_data_batches([image_bytes])
            pred = model.predict(data)
            pred_labels = get_pred_label(pred[0])

            data = {"prediction": pred_labels}
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)})
    return "OK"


if __name__ == "__main__":
    app.run(debug=True)
