import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
from PIL import Image

# Load the model
@st.cache_resource
def load_keras_model(model_path):
    model = load_model(model_path)
    return model

# Preprocess the image
def preprocess_image(image):
    target_size = (256, 256)
    # Resize image
    image = image.resize(target_size)
    # Convert image to array
    image_array = img_to_array(image)
    # Normalize image
    image_array = image_array / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Predict the class of the image
def predict_image(model, image):
    # Preprocess the image
    image_array = preprocess_image(image)
    # Make prediction
    prediction = model.predict(image_array)
    # Get the class with the highest probability
    class_index = np.argmax(prediction, axis=1)[0]
    return class_index

# Define class labels
class_labels = ['Tomato_Bacterial_spot', 'Corn_Common_rust', 'Potato_Early_blight']

# Streamlit app
def main():
    st.title("Plant Disease Classification")
    st.write("Upload an image of a plant leaf to classify its disease.")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Load model
        model_path = 'path/to/your/model/plant_disease.keras'  # Update with the correct path
        model = load_keras_model(model_path)

        # Predict
        predicted_class = predict_image(model, image)
        st.write(f"Predicted class index: {predicted_class}")
        st.write(f"Predicted class label: {class_labels[predicted_class]}")

if __name__ == "__main__":
    main()
