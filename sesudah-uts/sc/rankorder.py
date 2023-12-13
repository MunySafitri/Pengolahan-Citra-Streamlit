import streamlit as st
import numpy as np
import cv2
from PIL import Image

def apply_rank_filter(image, filter_size):
    # Convert BytesIO object to array
    img_array = np.array(image)

    # Convert to grayscale if the image is in color
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Apply the median filter
    filtered_img = cv2.medianBlur(img_array, filter_size)

    return filtered_img

def main():
    st.title("Rank-Order Filtering App")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Get user input for filter size
        filter_size = st.slider("Select filter size:", 3, 15, 5, step=2)

        # Apply the rank-order filter
        filtered_image = apply_rank_filter(image, filter_size)

        # Display the original and filtered images
        st.image([image, filtered_image], caption=["Original Image", "Filtered Image"], use_column_width=True)

if __name__ == "__main__":
    main()
