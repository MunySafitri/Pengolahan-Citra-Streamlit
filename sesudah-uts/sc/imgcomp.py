import streamlit as st
import base64
import io
from PIL import Image
import numpy as np

def compress_image(original_image, reduction_factor=64):
    # Convert the image to a numpy array
    img_array = np.array(original_image)

    # Apply lossy compression by rounding pixel values
    compressed_array = np.round(img_array / reduction_factor) * reduction_factor

    # Convert the compressed array back to an image
    compressed_image = Image.fromarray(compressed_array.astype('uint8'))

    # Convert the compressed image to binary data
    buffered = io.BytesIO()
    compressed_image.save(buffered, format="PNG")
    return buffered.getvalue()

def main():
    st.title("Lossy Image Compression App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        original_image = Image.open(uploaded_file)

        st.image(original_image, caption="Original Image", use_column_width=True)

        reduction_factor = st.slider("Color Reduction Factor", min_value=2, max_value=256, value=64)

        # Compress the image using the lossy compression method
        compressed_data = compress_image(original_image, reduction_factor=reduction_factor)

        # Save the compressed binary data to a file
        with open('compressed_data.bin', 'wb') as file:
            file.write(compressed_data)

        # Provide a download link for the binary compressed data
        st.markdown("### Download Compressed Data (Binary)")
        download_link = f'<a href="data:application/octet-stream;base64,{base64.b64encode(compressed_data).decode()}" ' \
                        f'download="compressed_data.bin">Download Compressed Data (Binary)</a>'
        st.markdown(download_link, unsafe_allow_html=True)

        # Display the compressed image
        compressed_image = Image.open(io.BytesIO(compressed_data))
        st.image(compressed_image, caption="Compressed Image", use_column_width=True)

if __name__ == "__main__":
    main()
