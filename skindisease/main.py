import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import io
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import numpy as np
from PIL import Image

from flask import Flask, request, jsonify

unique_labels = ['Acne and Rosacea', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 'Atopic Dermatitis', 'Bullous Disease', 'Cellulitis Impetigo and other Bacterial Infections', 'Eczema', 'Exanthems and Drug Eruptions', 'Hair Loss Photos Alopecia and other Hair Diseases', 'Herpes HPV and other STDs', 'Light Diseases and Disorders of Pigmentation', 'Lupus and other Connective Tissue diseases', 'Melanoma Skin Cancer Nevi and Moles', 'Nail Fungus and other Nail Disease', 'Poison Ivy Photos and other Contact Dermatitis', 'Psoriasis pictures Lichen Planus and related diseases', 'Scabies Lyme Disease and other Infestations and Bites', 'Seborrheic Keratoses and other Benign Tumors', 'Systemic Disease', 'Tinea Ringworm Candidiasis and other Fungal Infections', 'Urticaria Hives', 'Vascular Tumors', 'Vasculitis', 'Warts Molluscum and other Viral Infections']

IMG_SIZE = 480

def process_image(image):
  image = tf.image.decode_jpeg(image, channels = 3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, size = [IMG_SIZE, IMG_SIZE])
  return image

BATCH_SIZE = 32

def create_data_batches(X, batch_size = BATCH_SIZE):
  data = tf.data.Dataset.from_tensor_slices((tf.constant(X))) 
  data_batch = data.map(process_image).batch(BATCH_SIZE)
  return data_batch

def get_pred_label(prediction_probabilities):
  return unique_labels[np.argmax(prediction_probabilities)]

def load_model(model_path):
  print(f"Loading Saved Model From: {model_path}")
  model = tf.keras.models.load_model(model_path,
                                     custom_objects = {"KerasLayer": hub.KerasLayer})
  return model

model = load_model("model.h5")

# ************************************************************ #

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get('file')
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
