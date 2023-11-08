from flask import Flask, flash, request, redirect, url_for, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
import os


app = Flask(__name__)
app.debug = True
CORS(app)


UPLOAD_FOLDER = 'upload/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Load model
model = tf.keras.models.load_model(f'C:\Cyclone\SAVE_MODEL\save_cnn_model_1o4_batch=8_lr=1e3', compile=False)

@app.route('/', methods=['GET']) 
def Hello():
    try:
        return {"message": "Hello World", "status_code": 1, "status": "success"}
    except Exception as e:
        return {"message": "Facing some error :)", "status_code": 0, "status": "error"}


@app.route('/predict', methods=['POST'])
def Predict():
    if 'image' not in request.files:
        return {
            "status": "error",
            "status_code": 0,
            "message": "Upload an image first"
        }
    file = request.files['image']
    if file.filename == '':
        flash('No image selected for uploading')
        return {
            "status": "error",
            "status_code": 0,
            "message": "No image selected for uploading"
        }

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        # Save image 
        savePath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(savePath)

        # prediction
        result = model.predict(savePath)


        # Now code here :) i don't know what to do :)





if __name__ == '__main__':
    app.run(port=8000)