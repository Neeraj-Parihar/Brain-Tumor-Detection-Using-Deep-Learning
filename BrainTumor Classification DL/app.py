import os
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)


model =load_model('BrainTumor10Epochs.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

def is_brain_mri(img_path):
    try:
        # Read and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize if needed

        # Use the pre-trained model to predict the class
        prediction = brain_mri_classifier.predict(img_array)
        predicted_class = np.argmax(prediction, axis=-1)

        # Assuming class 1 corresponds to brain MRI
        return predicted_class == 1

    except Exception as e:
        print(f"Error during is_brain_mri: {e}")
        return False
def get_className(classNo):
	if classNo==0:
		return "No Brain Tumor"
	elif classNo==1:
		return "Yes Brain Tumor"


def getResult(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = model.predict(input_img)
    predicted_class = np.argmax(result, axis=-1)
    return result


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value = getResult(file_path)
        result = get_className(value)
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)