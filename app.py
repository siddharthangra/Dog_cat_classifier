from flask import Flask, request, jsonify, render_template
from tensorflow import keras
import numpy as np
import cv2
import io
import os

model = keras.models.load_model("model.h5")

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf


app = Flask(__name__)

def preprocess_image(image):
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def remove_transparency_channel(image):
    if image.shape[2] == 4:
        return image[:, :, :3]
    return image
 
@app.route('/')
def home():
    return render_template('Index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = remove_transparency_channel(image)
        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)

        label = "Dog" if prediction[0][0] > 0.5 else "Cat"
        
        return jsonify({'prediction': label})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)