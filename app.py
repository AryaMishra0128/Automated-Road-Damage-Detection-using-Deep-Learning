import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# Load model
model = YOLO("best.pt")  # Make sure best.pt is in same folder

# Title
st.title("🚧 Road Damage Detection Website")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded file to image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Convert image to numpy array for YOLO
    img_array = np.array(image)

    # Prediction
    results = model.predict(img_array)

    # Show predictions
    annotated_frame = results[0].plot()  # YOLOv8 plots predictions
    st.image(annotated_frame, caption='Prediction', use_column_width=True)
