import cv2
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
from skimage import exposure, io
#----------Image to display--------------

def warpaffine(image):
    rows, cols, ch = image.shape
    pts1 = np.float32([[50, 50],
                       [200, 50],
                       [50, 200]])
    pts2 = np.float32([[50, 100],
                       [200, 50],
                       [150, 200]])
    points = cv2.getAffineTransform(pts1, pts2)
    img = cv2.wrapAffine(image, points, (cols, rows))
    img_conv = Image.fromarray(img)
    return img_conv

#----------Gray Scale----------------
def gray_scale(image):
   
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

#------------Edge Detection-----------
def edge_detection(image, t1=30, t2=100):
    #Sebelum kita menerapkan detektor tepi Canny ke gambar, 
    ##kita perlu mengonversi gambar ke skala abu-abu dengan 
    #menggunakan fungsi cvtColor, kemudian memplot gambar 
    #dengan cara yang sama seperti yang kita lakukan pada gambar aslinya.
    #Fungsi canny memerlukan tiga hal: gambar skala abu-abu, nilai ambang piksel 
    #yang lebih rendah dan lebih tinggi untuk dipertimbangkan.

    edges = cv2.Canny(gray_scale(image), threshold1=t1, threshold2=t2)
    # edges = cv2.Canny(image, 40, 80, L2gradient=True)
    # edges = cv2.Canny(gray_scale(image), threshold1=30, threshold2=100)
    # img_conv = Image.fromarray(edges)
    return edges

#----------Blurred----------
def Blurred(image, kernel, option):
    # st.markdown(option)
    if option == "Mean Blur":
        Blur = Mean_Blur(image,kernel)
    elif option == "Gaussian Blur":
        Blur = Gaussian_Blur(image, kernel)
    elif option == "Median Blur":
       Blur = Median_Blur(image, kernel)
    else:
        st.markdown("Filter Belum Dipilih")
        return None
    # image boundary (BORDER_DEFAULT)
    # Blur = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)
    # img_conv = Image.fromarray(Blur)
    return Blur

#--------Gaussian Blur--------------

def Gaussian_Blur(image, kernel):
    # image boundary (BORDER_DEFAULT)
    Blur = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)
    img_conv = Image.fromarray(Blur)
    return img_conv

#--------Median Blur--------------

def Median_Blur(image, kernel):
    # image boundary (BORDER_DEFAULT)
    Blur = cv2.medianBlur(image, kernel)
    img_conv = Image.fromarray(Blur)
    return img_conv

#--------Mean Blur--------------

def Mean_Blur(image, kernel):
    # image boundary (BORDER_DEFAULT)
    Blur = cv2.blur(image, (kernel, kernel))
    img_conv = Image.fromarray(Blur)
    return img_conv

def equalizeHistogram(image):
    # image boundary (BORDER_DEFAULT)
    # img_conv = cv2.equalizeHist(gray_scale(image))
    #histogram equalization
    image_eq = exposure.equalize_hist(image)
    
    return image_eq












#----------Reduce noise----------

def reduce_noise(image):
    '''InputArray src,
	OutputArray dst,
	float h = 20,
	float hColor = 20,
	int templateWindowSize = 7,
	int searchWindowSize = 21'''
    #NON-LOCAL MEANS DENOISING ALGOROTHM
    noiseless_image_colored = cv2.fastNlMeansDenoisingColored(image, None, 20, 20, 7, 21)
    img_conv = Image.fromarray(noiseless_image_colored)
    return img_conv

#---------Negative Transformation----------

def negative_transformation(image):
    height, width, _ = image.shape
    for i in range(0, height - 1):
        for j in range(0, width - 1):
            pixel = image[i, j]
            pixel[0] = 255 - pixel[0]
            pixel[1] = 255 - pixel[1]
            pixel[2] = 255 - pixel[2]
    return image


#------------Sharping-------------------

def sharp_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    img_conv = Image.fromarray(image_sharp)
    return img_conv
