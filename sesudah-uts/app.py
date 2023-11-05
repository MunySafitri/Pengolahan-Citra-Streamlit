"""
# My first app
Here's our first attempt at using data to create a table:
"""
import streamlit as st
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np


#--------------implement Screen-------------
#--------Pertama Upload Gambar-----------------
#-----------Convert Gambar ke array----------
#---------Kedua Pilih Filter from Kotak-----
def Morfologi(image,option):
    if option == "Closing":
      # image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), -1)

      # Convert the image to grayscale
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

      # Apply binary thresholding to create a binary mask
      _, binary_mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

      # Define a kernel for the morphological operation (you can adjust the kernel size)
      kernel = np.ones((5, 5), np.uint8)

      # Apply morphological closing
      closing = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
      #jika ga ingin gambar binary
      # closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


      # Display the original image, binary mask, and segmented result
      st.image(image, caption="Original Image", use_column_width=True)
      #hapus jika tidak menggunakan binary
      st.image(binary_mask, caption="Binary Mask", use_column_width=True)
      ###
      st.image(closing, caption="Segmented Result (Morphological Closing)", use_column_width=True)
    
    elif option == "Dilation":
          # Convert the image to grayscale
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

      # Apply binary thresholding to create a binary mask
      _, binary_mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

      # Define a kernel for the morphological operation (you can adjust the kernel size)
      kernel = np.ones((5, 5), np.uint8)

      # Apply morphological dilation
      dilation = cv2.dilate(binary_mask, kernel, iterations=1)

      # Display the original image, binary mask, and segmented result
      st.image(image, caption="Original Image", use_column_width=True)
      st.image(binary_mask, caption="Binary Mask", use_column_width=True)
      st.image(dilation, caption="Segmented Result (Morphological Dilation)", use_column_width=True)
    
    elif option == "Edge":
         # Convert the image to grayscale
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
      
      # Apply binary thresholding to create a binary mask
      _, binary_mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)


      # Apply Canny edge detection
      edges = cv2.morphologyEx(binary_mask,cv2.MORPH_GRADIENT,kernel)  # You can adjust the thresholds (100 and 200) as needed

      # Display the original image and the segmented edges
      st.image(image, caption="Original Image", use_column_width=True)
      st.image(edges, caption="Segmented Edges (Canny)", use_column_width=True)

    elif option == "Erosion":
         # Convert the image to grayscale
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

      # Apply binary thresholding to create a binary mask
      _, binary_mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

      # Define a kernel for the morphological operation (you can adjust the kernel size)
      kernel = np.ones((5, 5), np.uint8)

      # Apply morphological erosion
      erosion = cv2.erode(binary_mask, kernel, iterations=1)

      # Display the original image, binary mask, and segmented result
      st.image(image, caption="Original Image", use_column_width=True)
      st.image(binary_mask, caption="Binary Mask", use_column_width=True)
      st.image(erosion, caption="Segmented Result (Morphological Erosion)", use_column_width=True)

    elif option == "Opening":
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
    
    elif option == "Translation":
      # Store height and width of the image 
      height, width = image.shape[:2] 
        
      quarter_height, quarter_width = height / 4, width / 4
        
      T = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]]) 
        
      # We use warpAffine to transform 
      # the image using the matrix, T 
      #fungsi untuk melakuakn translasi, rotasi dan operasi skala
      img_translation = cv2.warpAffine(image, T, (width, height)) 
       
       # Display the original image, binary mask, and segmented result
      st.image(image, caption="Original Image", use_column_width=True)
      # st.image(binary_mask, caption="Binary Mask", use_column_width=True)
      st.image(img_translation, caption="Translation Result", use_column_width=True)
   
    elif option == "Linear":
      scale_factor = st.slider("Scale Factor", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
      # Store height and width of the image 
      height, width, channels = image.shape
      new_height = int(height * scale_factor)
      new_width = int(width * scale_factor)

      # Perform linear interpolation using OpenCV's resize function
      interpolated_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

      # return interpolated_image
       
       # Display the original image, binary mask, and segmented result
      st.image(image, caption="Original Image", use_column_width=True)
      # st.image(binary_mask, caption="Binary Mask", use_column_width=True)
      st.image(interpolated_image, caption="Translation Result", use_column_width=True)
    
    elif option == "Bilinear":
      scale_factor = st.slider("Scale Factor", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
      # Store height and width of the image 
      height, width, channels = image.shape
      new_height = int(height * scale_factor)
      new_width = int(width * scale_factor)

      # Perform bilinear interpolation using OpenCV's resize function
      interpolated_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
       
       # Display the original image, binary mask, and segmented result
      st.image(image, caption="Original Image", use_column_width=True)
      # st.image(binary_mask, caption="Binary Mask", use_column_width=True)
      st.image(interpolated_image, caption="Translation Result", use_column_width=True)
   
    elif option == "Cubic":
      scale_factor = st.slider("Scale Factor", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
      # Store height and width of the image 
      height, width, channels = image.shape
      new_height = int(height * scale_factor)
      new_width = int(width * scale_factor)

      # Perform cubic interpolation using OpenCV's resize function
      interpolated_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

       # Display the original image, binary mask, and segmented result
      st.image(image, caption="Original Image", use_column_width=True)
      # st.image(binary_mask, caption="Binary Mask", use_column_width=True)
      st.image(interpolated_image, caption="Translation Result", use_column_width=True)


def main():
    st.header("Pengolahan Citra Digital")
    image_upload = st.file_uploader('Masukkan file gambar disini...',type=['jpg','png','jpeg'])
    if image_upload is not None:
        image = Image.open(image_upload)
        image_cv2 = np.array(image)
        # st.write(image_cv2)
#------------convert BGR to RGB-------------
        #membuat dan merencanakan gambar 
        #Untuk mendapatkan warna asli, kita perlu mengonversi 
        # warna ke format RGB menggunakan fungsi cvtColor dan 
        #menerapkannya pada gambar yang dimuat.

        # image_cv2_2 = image_cv2
        image_cv2 = cv2.cvtColor(image_cv2,cv2.COLOR_BGR2BGRA)
        # option = st.selectbox('Pilih Filter',('Pilih','Edge Detection','Grayscale','Negative Transformation','Gaussian Blur','Reduce Noise','Sharping'))
        option = st.selectbox('Pilih Filter',('Pilih','Morfologi'))
        st.write('Kamu memilih:',option)

#---------Choose from selectbox------------

        if option == 'Select':
            pass

#--------Edge Detection in selectionbox---
        elif option == 'Morfologi':
            # st.header('Gambar yang diinput')
            # st.image(image)
            option2 = st.selectbox('Pilih Filter',('Pilih','Closing','Dilation','Edge','Erosion','Opening',"Translation", "Linear", "Bilinear", "Cubic"))
            if option2 is not None:
                # st.image(image) 
                # result = Morfologi(image_cv2,option2)
                Morfologi(image_cv2,option2)
                # result = Blurred(image_cv2,input,option2)
                # if result is not None:
                #     st.markdown('Gambar setelah '+ option2)
                #     st.image(result, clamp=True)
        else:
            pass

if __name__ =="__main__":
    main()

