import streamlit as st
import cv2
import numpy as np

def linear_interpolation(image, scale_factor):
    height, width, channels = image.shape
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)

    # Perform linear interpolation using OpenCV's resize function
    interpolated_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return interpolated_image

def main():
    st.title("Image Linear Interpolation App")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    scale_factor = st.slider("Scale Factor", min_value=0.1, max_value=3.0, value=1.0, step=0.1)

    if uploaded_image is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), -1)
        st.image(uploaded_image, caption="Original Image", use_column_width=True)

        if st.button("Interpolate Image"):
            interpolated_image = linear_interpolation(image, scale_factor)
            st.image(interpolated_image, caption=f"Interpolated Image (Scale Factor: {scale_factor})", use_column_width=True)

if __name__ == "__main__":
    main()
