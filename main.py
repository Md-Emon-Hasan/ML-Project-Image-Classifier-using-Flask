from flask import Flask
from flask import render_template
from flask import request
import os
import logging
from werkzeug.utils import secure_filename
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import decode_predictions

app = Flask(__name__)

# configre logging
logging.basicConfig(level=logging.DEBUG)

# Initialize the model
model = ResNet50(weights='imagenet')

# Directory to store uploaded images
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR,'images')

# Ensure the images directory exists
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    if 'imagefile' not in request.files:
        return "No file part"

    imagefile = request.files['imagefile']

    if imagefile.filename == '':
        return "No selected file"

    # Ensure the filename is safe and save the file
    filename = secure_filename(imagefile.filename)
    image_path = os.path.join(IMAGE_DIR, filename)

    logging.debug(f'Saving image to: {image_path}')

    try:
        imagefile.save(image_path)

        # Load and preprocess the image
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)

        # Make predictions
        yhat = model.predict(image)
        label = decode_predictions(yhat)
        label = label[0][0]

        # Format the classification result
        classification = '%s (%.2f%%)' % (label[1], label[2] * 100)
    except Exception as e:
        return str(e)

    return render_template('index.html', prediction=classification)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Get port from environment variable
    app.run(host="0.0.0.0", port=port)
