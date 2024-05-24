import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import gradio as gr

# Load the entire model
loaded_model = tf.keras.models.load_model('cnn_model.h5')

# Define labels for classifying digits
labels = [str(i) for i in range(10)]

def predict(img):
    img = img.convert('L')  # convert to grayscale
    img = img.resize((28, 28))  # resize to 28x28
    img_array = image.img_to_array(img)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = loaded_model.predict(img_array)
    predicted_label = labels[np.argmax(prediction)]

    return predicted_label

# Create Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.components.Image(type='pil', label="Upload Image"),
    outputs=gr.components.Label(label="Predicted Digit")
)

# Launch the Gradio interface
if __name__ == "__main__":
    interface.launch(share=True)