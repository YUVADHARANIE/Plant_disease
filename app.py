import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image
import io

# Load the model
model_path = 'https://github.com/YUVADHARANIE/Plant_disease/blob/main/plant_disease%20(1).keras'
model = load_model(model_path)

# Class labels
class_labels = ['Tomato_Bacterial_spot', 'Corn_Common_rust', 'Potato_Early_blight']

# Image preprocessing
def preprocess_image(image):
    image = image.resize((256, 256))  # Resize image to match model input
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Prediction function
def predict_image(image):
    image_array = preprocess_image(image)
    prediction = model.predict(image_array)
    class_index = np.argmax(prediction, axis=1)[0]
    return class_index

# Streamlit app
st.title("Plant Disease Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Make prediction
    class_index = predict_image(image)
    st.write(f"Predicted class index: {class_index}")
    st.write(f"Predicted class label: {class_labels[class_index]}")
