from __future__ import division, print_function
import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import ReLU
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
from werkzeug.utils import secure_filename
from PIL import Image
import hashlib
from CustomLayers import RoutingLayer
from scipy.special import softmax
from CustomLayers import *
app = Flask(__name__, template_folder="templates") 
# Path to your model file
MODEL_PATH = 'model.h5'
cache = {}
def custom_sparse_categorical_crossentropy(**kwargs):
    config = kwargs.get('config', {})
    from_logits = config.get('from_logits', False)
    reduction = config.get('reduction', 'auto')
    name = config.get('name', 'sparse_categorical_crossentropy')
    return SparseCategoricalCrossentropy(from_logits=from_logits, reduction=reduction, name=name)

# Custom object dictionary
custom_objects = {
    'ReLU': ReLU,
    'SparseCategoricalCrossentropy': custom_sparse_categorical_crossentropy,
    'VggExtractor':VggExtractor,
    'StatsPooling':StatsPooling,
    'View':View,
    'PrimaryCapsule':PrimaryCapsule,
    'OutputCapsule':OutputCapsule,
    'RoutingLayer':RoutingLayer
}

try:
    model = load_model(MODEL_PATH, custom_objects=custom_objects)
    faces_model=load_model('FacesModel.h5',custom_objects=custom_objects)
except Exception as e:
    model = None
    faces_model=None
    print(f"Error loading model: {e}")

def model_predict(img_path, model,img_size):
    img = Image.open(img_path)
    img = img.convert("RGB")
    img = img.resize((img_size, img_size)) 
    x = image.img_to_array(img)
    x = x / 255.0 
    x=np.expand_dims(x,axis=0)
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
       
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'uploads')

        if not os.path.exists(upload_path):
            os.makedirs(upload_path)

        file_path = os.path.join(upload_path, secure_filename(f.filename))
        f.save(file_path)
        with open(file_path, 'rb') as img_file:
            file_hash = hashlib.md5(img_file.read()).hexdigest()
        cache_key = file_hash
        # Make prediction
        result = cache.get(cache_key)
        if result is None:
            preds = model_predict(file_path, model,200)
            predicted_class = np.argmax(preds, axis=1)[0]
            if predicted_class==0: result=f"the image is more like to be REAL!"
            else: result=f"the image is more like to be FAKE!"
            cache[cache_key] = result
        return render_template('upload.html', pred=result)
        
    return render_template('upload.html')

@app.route('/uploadFace', methods=['GET', 'POST'])
def uploadFace():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'uploads')

        if not os.path.exists(upload_path):
            os.makedirs(upload_path)

        file_path = os.path.join(upload_path, secure_filename(f.filename))
        f.save(file_path)
        with open(file_path, 'rb') as img_file:
            file_hash = hashlib.md5(img_file.read()).hexdigest()
        cache_key = file_hash
        # Make prediction
        result = cache.get(cache_key)
        if result is None:
            preds = model_predict(file_path, faces_model,150)
            predicted_class = np.argmax(preds, axis=1)[0]
            if predicted_class==0: result=f"the image is more like to be REAL!"
            else: result=f"the image is more like to be FAKE!"
            cache[cache_key] = result
        return render_template('uploadFace.html', pred=result)
        
    return render_template('uploadFace.html')

if __name__ == '__main__':
    app.run(debug=True)







