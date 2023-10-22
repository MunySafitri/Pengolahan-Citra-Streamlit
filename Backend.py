import cv2
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
from skimage import exposure, io, segmentation, color
import pandas as pd
import random, sys

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
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #Sebelum kita menerapkan detektor tepi Canny ke gambar, 
    ##kita perlu mengonversi gambar ke skala abu-abu dengan 
    #menggunakan fungsi cvtColor, kemudian memplot gambar 
    #dengan cara yang sama seperti yang kita lakukan pada gambar aslinya.
    #Fungsi canny memerlukan tiga hal: gambar skala abu-abu, nilai ambang piksel 
    #yang lebih rendah dan lebih tinggi untuk dipertimbangkan.

    edges = cv2.Canny(image = image_rgb, threshold1=t1, threshold2=t2)
    # edges = cv2.Canny(gray_scale(image_rgb), threshold1=t1, threshold2=t2)
   
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
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        face_roi = image[y:y+h, x:x+w]
        blurred = cv2.GaussianBlur(face_roi, (kernel, kernel), 0)
        image[y:y+h, x:x+w] = blurred
    # Blur = cv2.blur(image, (kernel, kernel))
    img_conv = Image.fromarray(image)
    return img_conv

#--------Median Blur--------------

def Median_Blur(image, kernel):
    # image boundary (BORDER_DEFAULT)
    # Blur = cv2.medianBlur(image, kernel)
    # img_conv = Image.fromarray(Blur)
    # return img_conv
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        face_roi = image[y:y+h, x:x+w]
        blurred = cv2.medianBlur(face_roi, kernel)
        image[y:y+h, x:x+w] = blurred
    # Blur = cv2.blur(image, (kernel, kernel))
    img_conv = Image.fromarray(image)
    return img_conv

#--------Mean Blur--------------

def Mean_Blur(image, kernel):
    # image boundary (BORDER_DEFAULT)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        face_roi = image[y:y+h, x:x+w]
        blurred = cv2.blur(face_roi, (kernel, kernel))
        image[y:y+h, x:x+w] = blurred
    # Blur = cv2.blur(image, (kernel, kernel))
    img_conv = Image.fromarray(image)
    return img_conv

def equalizeHistogram(image):
    # image boundary (BORDER_DEFAULT)
    # img_conv = cv2.equalizeHist(gray_scale(image))
    #histogram equalization
    image_eq = exposure.equalize_hist(image)
    
    return image_eq

def Segmentasi(image, option):
    # st.markdown(option)
    if option == "Bitwise":
        st.markdown('Batas Bawah, ukuran HSV')
        st.markdown('HUE')
        h1 =  st.select_slider("Hue bawah",range(0,256), value=30)
        st.markdown('Saturation')
        s1 =  st.select_slider("Saturation bawah",range(0,256), value=50)
        st.markdown('Value')
        v1 =  st.select_slider("Value bawah",range(0,256), value=50)
        st.markdown('Batas Atas, ukuran HSV')
        st.markdown('HUE')
        h2 =  st.select_slider("Hue atas",range(0,256), value=100)
        st.markdown('Saturation')
        s2 =  st.select_slider("Saturation atas",range(0,256), value=255)
        st.markdown('Value')
        v2 =  st.select_slider("Value atas",range(0,256), value=255)
        
        Segmentasi = bitwise(image,h1,h2,s1,s2,v1,v2)
    elif option == "Thresholding":
        Segmentasi = threshold(image)
    elif option == "Morfologi":
        option2 = st.selectbox('Pilih Filter',('Pilih','Closing(Binary)','Erosi','Dilasi'))
        if option2 == "Closing(Binary)": 
            Segmentasi = closingBinary(image,)
        elif option2 == "Erosi":
            Segmentasi = Erosion(image)
        elif option2 == "Dilasi":
            Segmentasi = Dilation(image)
            # st.markdown(Segmentasi)
        else :
            return None
        # return Segmentasi
    else:
        st.markdown("Filter Belum Dipilih")
        return None
   
    return Segmentasi

#deteksi ijo atau kuning
def bitwise(image,h1,h2,s1,s2,v1,v2):
    # face_cascade = cv2.CascadeClassifier("face_detector.xml")
    if image is None:
        st.markdown("Gagal membaca gambar")
    else:
        # Konversi citra dari BGR ke HSV (Hue, Saturation, Value)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Tentukan batas-batas warna yang akan disegmentasi
    lower_color = np.array([h1, s1, v1])  # Nilai batas bawah dalam format HSV
    upper_color = np.array([h2, s2, v2])  # Nilai batas atas dalam format HSV

        # Buat mask untuk citra berdasarkan batas warna (hijau/kuning)
    mask = cv2.inRange(hsv_image, lower_color, upper_color)

        # Lakukan operasi bitwise AND untuk memotong objek dari citra asli
    segmented_image = cv2.bitwise_and(image, image, mask=mask)
    # cv2.imshow('Citra Segmentasi', segmented_image)

    return segmented_image

def threshold(image):
    if image is None:
        st.markdown("Gagal membaca gambar")
    else:
        # Konversi citra ke citra grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Terapkan ambang batas (thresholding) untuk segmentasi
        _, segmented_image = cv2.threshold(gray_image, 125, 255, cv2.THRESH_BINARY)
        return segmented_image

 #Closing Binary
def closingBinary(image):
    #original image
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    st.markdown("konversi ke RGB")
    st.image(image)

    #grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    st.markdown("Konversi ke abu-abu")
    
    st.image(gray)

    #threshold image
    ret, thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV +cv2.THRESH_OTSU)
    st.markdown("Menggunakan thresholding")
    st.image(thresh)

    #segmented image
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,kernel, iterations = 15)
    # closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,kernel, iterations = 15)
    bg = cv2.dilate(closing, kernel, iterations = 1)
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2,0)
    ret, fg = cv2.threshold(dist_transform, 0.02*dist_transform.max(), 255, 0)
    st.markdown("hasil pendekatan Euclidean")
    # st.image(fg,clamp=True)
    return fg

#----------Erosi Morphology----------    

def Erosion(image):
    # Mengambil matriks dengan ukuran 5 sebagai kernel
    iterasi =  st.select_slider("iterasi",range(0,15), value=1)

    kernel = np.ones((5, 5), np.uint8)
    #Parameter pertama adalah gambar asli,
# Kernel adalah matriks yang digunakan untuk mengkonvolusi gambar
# konvolusi dan parameter ketiga adalah jumlah
# iterasi, yang akan menentukan berapa banyak
# Anda ingin mengikis/memperkecil gambar yang diberikan.
    img_erosion = cv2.erode(image, kernel, iterations=iterasi)
    return img_erosion
#----------Erosi Morphology----------    

def Dilation(image):
    # Mengambil matriks dengan ukuran 5 sebagai kernel
    iterasi =  st.select_slider("iterasi",range(0,15), value=1)

    kernel = np.ones((5, 5), np.uint8)
    #Parameter pertama adalah gambar asli,
# Kernel adalah matriks yang digunakan untuk mengkonvolusi gambar
# konvolusi dan parameter ketiga adalah jumlah
# iterasi, yang akan menentukan berapa banyak
# Anda ingin mengikis/memperkecil gambar yang diberikan.
    img_dilation = cv2.dilate(image, kernel, iterations=iterasi)# Memplot gambar sumber dan gambar tujuan
    return img_dilation



def Resize(image,panjang,lebar):
    # st.markdown("resize")
    
    image = cv2.resize(image, ( lebar,panjang))
    return image

    # Mengambil matriks dengan ukuran 5 sebagai kernel
    
#----------Beauty Mode----------
#https://medium.com/swlh/how-i-implemented-my-own-augmented-reality-beauty-mode-3bf3b74e5507
def Beauty(image):
    # image = cv2.imread(image,-1)
    blur = cv2.bilateralFilter(image,28,45,45)

    hpf = image - blur + 128
    hpf = cv2.GaussianBlur(hpf,(5,5),0)

    img2 = image.astype(float)/255
    hpf2 = hpf.astype(float)/255 # make float on range 0-1

    mask = img2 >= 0.5 # generate boolean mask of everywhere a > 0.5
    ab = np.zeros_like(img2) # generate an output container for the blended image

    # now do the blending
    ab[~mask] = (2*img2*hpf2)[~mask] # 2ab everywhere a<0.5
    ab[mask] = (1-2*(1-img2)*(1-hpf2))[mask] # else this

    result =(ab*255).astype(np.uint8) #convert to cv22 format

    hpfInv = (255-hpf)

    img3 = image.astype(float)/255
    hpf3 = hpfInv.astype(float)/255 # make float on range 0-1

    mask = img3 >= 0.5 # generate boolean mask of everywhere a > 0.5
    ab2 = np.zeros_like(img3) # generate an output container for the blended image

    # now do the blending
    ab2[~mask] = (2*img3*hpf3)[~mask] # 2ab everywhere a<0.5
    ab2[mask] = (1-2*(1-img3)*(1-hpf3))[mask] # else this


    result2 =(ab2*255).astype(np.uint8)
    return result2

def binarize_image(img, threshold):
    # Ensure the image is grayscale
    if len(img.shape) == 3:
        img = np.mean(img, axis=-1)
    
    # Create an empty binary image
    binary_img = np.zeros_like(img)
    
    # Binarize the image
    binary_img[img > threshold] = 255
    
    return binary_img


#convulution
def convolution(input_img, kernel):
  #inisialisasi output dengan matrix dengan nilai 0
  output = np.zeros(input_img.shape)
  #padding angka 0 pada border input gambar
  padding = np.pad(input_img, pad_width=1, mode='constant', constant_values=0)
  for i in range(input_img.shape[0]):
    for j in range(input_img.shape[1]):
      for k in range(kernel.shape[0]):
        for l in range(kernel.shape[1]):
          output[i][j] = output[i][j] + kernel[k][l]*padding[i+k][j+l]
  
  return output

#thresholding untuk pixel sorting
def thresholding(input_img, threshold_value):
  output = np.zeros(input_img.shape)
  for i in range(0, output.shape[0]):
    for j in range(0, output.shape[1]):
      if (input_img[i][j] <= threshold_value):
        output[i][j] = 255
      else:
        output[i][j] = 0
  return output


# fungsi untuk melakukan sorting pada interval pixel dengan metode quick sort secara ascending
def quick_sort(pixels):
	if pixels == []:
		return pixels

	else:
		pivot = pixels[0]
		lesser = quick_sort([x for x in pixels[1:] if (x[0] + x[1] + x[2]) < (pivot[0] + pivot[1] + pivot[2])])
		greater = quick_sort([x for x in pixels[1:] if (x[0] + x[1] + x[2]) >= (pivot[0] + pivot[1] + pivot[2])])
		return lesser + [pivot] + greater

# fungsi untuk melakukan konversi dari data pixel (RGBA) menjadi Luminance/intensitas
def RGB_to_Luminance(data):
  # mengubah Color Model RGB menjadi Luminance/intensitas (0.2126*Red + 0.7152*Green + 0.0722*Blue) 
  return 0.2126*data[0] + 0.7152*data[1] + 0.0722*data[2]


def sort_all_pixels(img):
	# melakukan sorting terhadap semua pixel yang ada pada setiap baris
	# img = Image.open(image)
	# img = np.array(img)
	img = img.convert('RGBA')
	# image.append((Image.fromarray
    #             (img, dtype = 
    #              np.float32).convert('RGBA')/255.))
 	# img = image
    # img = np.array(img)
	data = img.load()
	new = Image.new('RGBA', img.size)

	pixels = []
	sorted_pixels = []
	# menyimpan data semua pixel ke list 2D (pixels)
	for y in range(img.size[1]):
		pixels.append([])
		for x in range(img.size[0]):
			pixels[y].append(data[x, y])

	for y in range(img.size[1]):
			sorted_pixels.append(quick_sort(pixels[y]))

	for y in range(img.size[1]):
		for x in range(img.size[0]):
			new.putpixel((x, y), sorted_pixels[y][x])

	new.save('output-sortall.png')





def pixel_sort_baris(img, threshold):
#   img = Image.open(image)
  # img = img.resize((img.width//2, img.height//2)) # memperkecil ukuran image
  img = img.convert('RGBA')
  data = img.load()
  new = Image.new('RGBA', img.size)

  pixels = []
  sorted_pixels = []

  for y in range(img.size[1]):
    pixels.append([])
    for x in range(img.size[0]):
      pixels[y].append(data[x, y])
  
  # membagi tiap baris menjadi beberapa interval dan mengurutkan setiap interval berdasarkan intensitasnya
  for y in range(img.size[1]):
    minsort = 0
    maxsort = 0
    sort = []
    while(maxsort < img.size[0]-1):
      # 2 buah pixel yang bersebelahan (horizontal) berada pada satu interval yang sama jika selisih intensitasnya kurang dari threshold
      if(abs(RGB_to_Luminance(pixels[y][maxsort]) - RGB_to_Luminance(pixels[y][maxsort+1])) < threshold):
        maxsort += 1
      else:
        maxsort += 1
        for x in range(minsort, maxsort):
          sort.append(pixels[y][x])
        # melakukan sorting pada interval pixels dengan algoritma quicksort
        sort = quick_sort(sort)
        
        # meng-assign interval pixels yang sudah terurut ke 2D list pixels
        i=0
        for x in range(minsort, maxsort):
          pixels[y][x] = sort[i]
          i = i + 1
        
        sort = []
        minsort = maxsort

    sorted_pixels.append(pixels[y])
  
  for y in range(img.size[1]):
    for x in range(img.size[0]):
      new.putpixel((x,y), sorted_pixels[y][x])
  
  new.save('output-sortIntervalsRow.png')

#---------------------pixelSorting-------------------
def pixel_sort_kolom(img, threshold):
#   img = Image.open(image)
  # img = img.resize((img.width//2, img.height//2)) # memperkecil ukuran image
  img = img.convert('RGBA')
  data = img.load()
  new = Image.new('RGBA', img.size)

  pixels = []
  sorted_pixels = []

  for y in range(img.size[1]):
    pixels.append([])
    for x in range(img.size[0]):
      pixels[y].append(data[x, y])
  
  # membagi tiap kolom menjadi beberapa interval dan mengurutkan setiap interval berdasarkan intensitasnya
  for y in range(img.size[0]):
    minsort = 0
    maxsort = 0
    sort = []
    while(maxsort < img.size[1]-1):
      # 2 buah pixel yang bersebelahan (vertikal) berada pada satu interval yang sama jika selisih intensitasnya kurang dari threshold
      if(abs(RGB_to_Luminance(pixels[maxsort][y]) - RGB_to_Luminance(pixels[maxsort+1][y])) < threshold):
        maxsort += 1
      else:
        maxsort += 1
        for x in range(minsort, maxsort):
          sort.append(pixels[x][y])
        # melakukan sorting pada interval pixels dengan algoritma quicksort
        sort = quick_sort(sort)
        
        # meng-assign interval pixels yang sudah terurut ke 2D list pixels
        i=0
        for x in range(minsort, maxsort):
          pixels[x][y] = sort[i]
          i = i + 1
        
        sort = []
        minsort = maxsort
  
  for y in range(img.size[1]):
    sorted_pixels.append(pixels[y])
  
  for y in range(img.size[1]):
    for x in range(img.size[0]):
      new.putpixel((x,y), sorted_pixels[y][x])
  
  new.save('output-sortIntervalsColumn.png')

def pixelSorting(image):
    #konversi ke grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #melakukan sharpening
    #Gaussian Blur
    base_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    constant = 1/16
    blur_kernel = constant*base_kernel
    #Standard sharpening
    n_kernel = np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]])
    sharpen_kernel = n_kernel - blur_kernel

    image_gray = convolution(image_gray, sharpen_kernel)
    
    # melakukan thresholding
    image_segmented = thresholding(image_gray,100)
    st.header('hasil thresholding')
    st.image(image_segmented, clamp=True)

    # st.header('hasil sort image')
    # #melakukan pixel sorting
    # sort_all_pixels(image)
    # resultimage = cv2.imread('output-sortall.png')
    # st.header('hasil sort all pixel')
    # st.image(resultimage)
# cv2_imshow(image_gray)




















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
