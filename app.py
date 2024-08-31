import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('plant_disease.h5')  # Ensure the model path is correct

# Print the model summary to check input shape (remove this line in production)
model.summary()

# Get the input shape from the model
input_shape = model.input_shape[1:3]  # Extracting height and width

# Define class names
class_names = ['Tomato_Bacterial_spot', 'Corn_Common_rust', 'Potato_Early_blight']  # Adjust based on your model's classes

# Define the function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize(input_shape)  # Resize image to match model input size
    img_array = np.array(img) / 255.0  # Normalize the image to [0, 1] range
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict class
def predict_image(img):
    img_preprocessed = preprocess_image(img)
    predictions = model.predict(img_preprocessed)
    predicted_class_index = np.argmax(predictions, axis=1)[0]  # Extract the index of the highest probability
    return class_names[predicted_class_index], predictions[0]

# Define the Streamlit app
st.title("Plant Disease Classification")
st.write("Upload an image of a plant leaf to get a disease prediction.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    img = Image.open(uploaded_file)
    predicted_class, prediction_probs = predict_image(img)
    
    # Display the result
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write(f'**Predicted Class:** {predicted_class}')
    st.write(f'**Prediction Probabilities:** {prediction_probs}')
