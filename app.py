import os
from werkzeug.utils import secure_filename
import cv2
from flask_cors import CORS
from flask import Flask, jsonify, request, render_template, redirect, url_for
import tensorflow as tf

app = Flask(__name__)
app.debug = True
CORS(app)
app.config['UPLOAD_FOLDER'] = 'upload/'
app.config['STATIC_FOLDER'] = "static/"
ALLOWED_EXTENSIONS = ['jpeg']
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load model
model = tf.keras.models.load_model(r'.\save_cnn_model_1o4_batch=8_lr=0.001\save_cnn_model_1o4_batch=8_lr=0.001', compile=False)

@app.route('/', methods=['GET', 'POST']) 
def Hello():
    if request.method == 'POST':
        if 'image' not in request.files:
            data = jsonify({
                "status": "error",
                "status_code": 404,
                "message": "Upload an image first"
            })
            return render_template('home.html', data=data)
        file = request.files['image']
        if file.filename == '':
            data = jsonify({
                "status": "error",
                "status_code": 404,
                "message": "No image selected for uploading"
            })
            return render_template('home.html', data=data)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Save image 
            savePath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(savePath)
            return redirect(url_for('PredictIntensity', path=savePath))
        else:
            data = jsonify({
                "status": "error",
                "status_code": 404,
                "message": "Unsupported image type"
            })
            return render_template('home.html', data=data)
    return render_template('home.html')
    
@app.route('/predict-intensity', methods=['GET', 'POST'])
def PredictIntensity():
    path = request.args.get("path")
    # Read image
    img = cv2.imread(path)
    img = cv2.resize(img, (128,128))
    x_true = tf.convert_to_tensor(img, tf.float32)
    x_true = tf.expand_dims(x_true, axis=0)
    # Normalize image
    x_true = x_true / 255.0 
    # prediction
    y_pred = model.predict(x_true)
    y_pred_arg = tf.math.argmax(y_pred, axis=-1) ## argmax
    func = lambda x: round(x*100,2)
    data = {
        "prediction":enumerate(map(func, y_pred[0])),
        "status_code": 200,
        "path": path,
        "result": int(y_pred_arg[0]+1)
    }
    return render_template('home.html', data=data)


if __name__ == '__main__':
    app.run(port=8000, debug=True)
