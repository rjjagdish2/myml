import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the entire model
loaded_model = load_model('cnn_model.h5')

# Define labels for classifying digits
labels = [str(i) for i in range(10)]

# Function to preprocess input image
def preprocess_image(img):
    img = img.convert('L').resize((28, 28))
    img_array = image.img_to_array(img)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Initialize Flask app
app = Flask(__name__)

loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Check if request contains file
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']

    # Check if file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Open the image file
        img = Image.open(file)
        # Preprocess the image
        img_array = preprocess_image(img)
        # Make prediction
        prediction = loaded_model.predict(img_array)
        # Get the predicted digit
        predicted_digit = labels[np.argmax(prediction)]
        return jsonify({'prediction': predicted_digit})
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

# if __name__ == '__main__':
#     app.run(debug=True)
