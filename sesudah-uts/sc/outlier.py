import streamlit as st
import numpy as np
import cv2
from PIL import Image

def apply_outlier_detection(image, threshold):
    # Convert BytesIO object to array
    img_array = np.array(image)

    # Convert to grayscale if the image is in color
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Calculate the median absolute deviation (MAD)
    mad = np.median(np.abs(img_array - np.median(img_array)))

    # Create a binary mask for outliers
    outlier_mask = np.abs(img_array - np.median(img_array)) > threshold * mad

    # Replace outliers with the median value
    img_array[outlier_mask] = np.median(img_array)

    return img_array

def main():
    st.title("Outlier Detection App")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Get user input for threshold
        threshold = st.slider("Select outlier threshold:", 0.1, 2.0, 1.0, step=0.1)

        # Apply the outlier detection
        outlier_filtered_image = apply_outlier_detection(image, threshold)

        # Display the original and filtered images
        st.image([image, outlier_filtered_image], caption=["Original Image", "Outlier Filtered Image"], use_column_width=True)

if __name__ == "__main__":
    main()
