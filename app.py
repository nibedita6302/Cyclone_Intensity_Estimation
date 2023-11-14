from flask import Flask, flash, request, redirect, url_for, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import os
import cv2


app = Flask(__name__)
app.debug = True
CORS(app)


UPLOAD_FOLDER = 'upload/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = 'jpg'
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Load model
model = tf.keras.models.load_model(r'.\save_cnn_model_1o4_batch=8_lr=0.001\save_cnn_model_1o4_batch=8_lr=0.001', compile=False)

@app.route('/', methods=['GET']) 
def Hello():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def Predict():
    if 'image' not in request.files:
        return {
            "status": "error",
            "status_code": 404,
            "message": "Upload an image first"
        }
    file = request.files['image']
    if file.filename == '':
        flash('No image selected for uploading')
        return {
            "status": "error",
            "status_code": 404,
            "message": "No image selected for uploading"
        }

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        # Save image 
        savePath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(savePath)
        # Read image
        img = cv2.imread(savePath)
        x_true = tf.convert_to_tensor(img, tf.float32)
        x_true = tf.expand_dims(x_true, axis=0)
        #print(img.shape)
        # Normalize image
        x_true = x_true / 255.0
        # prediction
        y_pred = model.predict(x_true)
        y_pred_arg = tf.math.argmax(y_pred, axis=-1) ## argmax
        return {
            'y_pred':y_pred,
            'y_pred_arg':y_pred_arg,
            'status_code': 200
        }
    
    else:
        return {
            "status": "error",
            "status_code": 415,
            "message": "Unsupported Media Type, server cannot process the request body."
        }


if __name__ == '__main__':
    app.run(port=8000, debug=True)