import cv2
import numpy as np
import streamlit as st

st.title("Image Segmentation using Morphological Opening")

# Upload an image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), -1)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding to create a binary mask
    _, binary_mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Define a kernel for the morphological operation (you can adjust the kernel size)
    kernel = np.ones((5, 5), np.uint8)

    # Apply morphological opening
    opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Display the original image, binary mask, and segmented result
    st.image(image, caption="Original Image", use_column_width=True)
    st.image(binary_mask, caption="Binary Mask", use_column_width=True)
    st.image(opening, caption="Segmented Result (Morphological Opening)", use_column_width=True)
