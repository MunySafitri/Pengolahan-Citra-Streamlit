import cv2
import numpy as np
import streamlit as st

st.title("Image Segmentation using Canny Edge Detection")

# Upload an image
uploaded_image = st.file_uploader("Choose an image..", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), -1)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)  # You can adjust the thresholds (100 and 200) as needed

    # Display the original image and the segmented edges
    st.image(image, caption="Original Image", use_column_width=True)
    st.image(edges, caption="Segmented Edges (Canny)", use_column_width=True)
