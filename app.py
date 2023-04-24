from flask import Flask, render_template, request
import os
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the saved models
glaucoma_model = tf.keras.models.load_model('glaucoma_model.h5')
cataract_model = tf.keras.models.load_model('cataract_model.h5')
diabetic_retinopathy_model = tf.keras.models.load_model('retinopathy_model.h5')

# Define the allowed file extensions for uploaded images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    """Check if a file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict(image_path, model):
    """Make a prediction on a fundus image using a given model"""
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(512, 512))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds[0][0]


@app.route('/')
def index():
    """Render the index page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_disease():
    """Make a prediction on an uploaded fundus image"""
    # Get the uploaded file
    file = request.files['file']

    # Check if file is empty
    if not file:
        return render_template('index.html', message='No file was uploaded.')

    # Check if the file has an allowed extension
    if not allowed_file(file.filename):
        return render_template('index.html', message='Invalid file type. Only PNG and JPEG files are allowed.')

    # Save the file to the uploads folder
    filename = secure_filename(file.filename)
    file_path = os.path.join('static/uploads', filename)
    file.save(file_path)

    # Determine which model to use based on user input
    model = None
    disease = request.form.get('disease')
    print(disease)
    if disease == 'glaucoma':
        model = glaucoma_model
    elif disease == 'cataract':
        model = cataract_model
    elif disease == 'retinopathy':
        model = diabetic_retinopathy_model

    # Make a prediction using the selected model
    if model:
        preds = predict(file_path, model)
        if preds < 0.9:
            disease_found = disease
            disease = "No Diseases Found"
        else:
            disease_found = disease
        predicted_prob = round(preds, 2)
        return render_template('result.html', preds=predicted_prob, disease=disease, disease_found=disease_found, filename=filename)
    else:
        return render_template('index.html', message='Invalid disease selected.')


if __name__ == '__main__':
    app.run(debug=True)