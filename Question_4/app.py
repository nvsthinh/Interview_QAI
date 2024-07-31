from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io
from utils import *
from model import MLP, config
app = Flask(__name__)

# Init model from weight
model_init = MLP(config.INPUT_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)
model = model_init.load('checkpoint/model.pkl')

def preprocess_image(image):
    # Convert the image to grayscale
    image = image.convert('L')  # Convert image to grayscale
    # Resize the image to 28x28 pixels for MNIST
    image = image.resize((28, 28))  # Resize to 28x28 for MNIST
    # Convert image to numpy array and normalize
    image_array = np.array(image) / 255.0  # Convert to numpy array and normalize
    # Flatten the array to match MLP input format
    image_array = image_array.flatten()  # Flatten to fit MLP input format
    return image_array

def inference(model, input):
    # Load the data
    label_features_list, unique_labels = load_pickle('checkpoint/label_features_list.pkl'), load_pickle('checkpoint/unique_labels.pkl')

    # Extract features from the input sample
    input_features = model.forward(input)

    # Calculate similarity scores with each label feature
    similarities = [cosine_similarity(input_features, label_feature) for label_feature in label_features_list]
    
    # Find the label with the highest similarity score
    predicted_label = unique_labels[np.argmax(similarities)]
    
    return predicted_label

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith(('.png', '.jpg', '.jpeg')):
        image = Image.open(io.BytesIO(file.read()))
        image_array = preprocess_image(image)
        prediction = inference(model, image_array)  # Dự đoán với mô hình Numpy
        predicted_label = int(prediction)
        return jsonify({'label': predicted_label})

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=False)