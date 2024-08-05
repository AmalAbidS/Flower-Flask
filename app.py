from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Load the model
model_path = 'model/my_model.h5'
if os.path.exists(model_path):
    print("Model file found.")
    model = tf.keras.models.load_model(model_path)
else:
    print("Model file not found.")
    model = None

# Class labels (adjust according to your classes)
class_labels = [
    'Adathoda',
    'Banana',
    'Bush Clock Vine',
    'Champaka',
    'Chitrak',
    'Common Lanthana',
    'Crown Flower',
    'Datura',
    'Four O\'Clock Flower',
    'Hibiscus',
    'Honeysuckle',
    'Indian Mallow',
    'Jatropha',
    'Malabar Melastome',
    'Marigold',
    'Nagapoovu',
    'Nityakalyani',
    'Pinwheel Flower',
    'Rose',
    'Shankupushpam',
    'Spider Lily',
    'Sunflower',
    'Thechi',
    'Thumba',
    'Touch Me Not',
    'Tridax Procumbens',
    'Wild Potato Vine',
    'Yellow Daisy'
]
def predict_image(img_path):
    if model is None:
        return "Model not loaded", 0

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Debugging: Print image array details
    print("Image Array Shape:", img_array.shape)
    print("Image Array Min/Max Values:", img_array.min(), img_array.max())

    predictions = model.predict(img_array)

    # Debugging: Print predictions
    print("Predictions:", predictions)

    predicted_class_index = np.argmax(predictions[0])

    # Ensure class index is valid
    if predicted_class_index < len(class_labels):
        predicted_class_label = class_labels[predicted_class_index]
        confidence = predictions[0][predicted_class_index]
    else:
        predicted_class_label = "Unknown"
        confidence = 0

    return predicted_class_label, confidence


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join('static/images', file.filename)
            file.save(file_path)
            label, confidence = predict_image(file_path)
            return render_template('index.html', label=label, confidence=confidence, image_path=file.filename)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('static/images', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

