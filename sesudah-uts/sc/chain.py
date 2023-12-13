import streamlit as st
import cv2
import numpy as np

def chain_code(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection (you can use a more sophisticated method based on your needs)
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours in the image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Chain code calculation (just a simple example here)
    chain_code_result = []
    for contour in contours:
        for point in contour:
            x, y = point[0]
            chain_code_result.append((x, y))
    
    return chain_code_result

def main():
    st.title("Chain Code Streamlit App")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

        # Display the original image
        st.image(image, caption="Original Image", use_column_width=True)

        # Apply chain code
        chain_code_result = chain_code(image)

        # Display the chain code result
        st.write("Chain Code Result:")
        st.write(chain_code_result)

if __name__ == "__main__":
    main()
