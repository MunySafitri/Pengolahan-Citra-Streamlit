"""
# My first app
Here's our first attempt at using data to create a table:
"""
import streamlit as st
from PIL import Image,ImageDraw
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack
from itertools import chain
import io
import heapq
import huffman
import base64
# import lzw

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

def Geometri(image,option):
    if option == "Outlier":
      st.header("Outlier")
  
        # Get user input for threshold
      threshold = st.slider("Select outlier threshold:", 0.1, 2.0, 1.0, step=0.1)
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


        # Display the original and filtered images
      st.image([image, img_array], caption=["Original Image", "Outlier Filtered Image"], use_column_width=True)

    elif option == "Rank Order":
      st.header("Rank Order")
       # Get user input for filter size
      filter_size = st.slider("Select filter size:", 3, 15, 5, step=2)
       # Convert BytesIO object to array
      img_array = np.array(image)

      # Convert to grayscale if the image is in color
      if len(img_array.shape) == 3:
          img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

      # Apply the median filter
      filtered_img = cv2.medianBlur(img_array, filter_size)
       # Display the original and filtered images
      st.image([image, filtered_img], caption=["Original Image", "Filtered Image"], use_column_width=True)
    
    elif option == "Low Pass Filter":
      st.header("Low Pass Filter")
      c_x = st.slider("size circle x:", 0, 200, 50, step=5)
      c_y = st.slider("size circle y:", 0, 200, 50, step=5)
      
#convert image to numpy array
      image1_np=np.array(image)
      if len(image1_np.shape) == 3:
          image1_np = cv2.cvtColor(image1_np, cv2.COLOR_RGB2GRAY)


      #fft of image
      fft1 = fftpack.fftshift(fftpack.fft2(image1_np))

      #Create a low pass filter image
      x,y = image1_np.shape[0],image1_np.shape[1]
      #size of circle
      e_x,e_y=c_x,c_y
      #create a box 
      bbox=((x/2)-(e_x/2),(y/2)-(e_y/2),(x/2)+(e_x/2),(y/2)+(e_y/2))

      low_pass=Image.new("L",(image1_np.shape[0],image1_np.shape[1]),color=0)

      draw1=ImageDraw.Draw(low_pass)
      draw1.ellipse(bbox, fill=1)

      low_pass_np=np.array(low_pass)

      #multiply both the images
      filtered=np.multiply(fft1,low_pass_np)

      #inverse fft
      ifft2 = np.real(fftpack.ifft2(fftpack.ifftshift(filtered)))
      ifft2 = np.maximum(0, np.minimum(ifft2, 255))
      st.image([image, ifft2.astype(np .uint8)], caption=["Original Image", "Filtered Image"], use_column_width=True)
   
    elif option == "Filter Median":
      st.header("Filter Median")
      image=np.array(image)
      if len(image.shape) == 3:
          image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

      # rows, cols = image.shape[:2]
      ksize = st.slider("Select filter size:", 3, 15, 6, step=3)
      padsize = int((ksize-1)/2)
      pad_img = cv2.copyMakeBorder(image, *[padsize]*4, cv2.BORDER_DEFAULT)
      temp = []
      indexer = ksize // 2
      for i in range(len(pad_img)):
          for j in range(len(pad_img)):
              for z in range(ksize):
                  if i + z - indexer < 0 or i + z - indexer > len(pad_img) - 1:
                      for c in range(ksize):
                          temp.append(0)
                  else:
                      if j + z - indexer < 0 or j + indexer > len(pad_img) - 1:
                          temp.append(0)
                      else:
                          for k in range(ksize):
                              temp.append(pad_img[i + z - indexer][j + k - indexer])
              temp.sort()
              pad_img[i][j] = temp[len(temp) // 2]
              temp = []
      # geomean = np.exp(cv2.boxFilter(np.log(pad_img), -1, (ksize, ksize)))
      st.image([image, pad_img.astype(np .uint8)], caption=["Original Image", "Filtered Image"], use_column_width=True)

# Python3 code for generating 8-neighbourhood chain
# code for a 2-D line
 
codeList = [5, 6, 7, 4, -1, 0, 3, 2, 1]
 
# This function generates the chaincode 
# for transition between two neighbour points
def getChainCode(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    hashKey = 3 * dy + dx + 4
    return codeList[hashKey]
 
# '''This function generates the list of 
# chaincodes for given list of points'''
def generateChainCode(ListOfPoints):
    chainCode = []
    for i in range(len(ListOfPoints) - 1):
        a = ListOfPoints[i]
        b = ListOfPoints[i + 1]
        chainCode.append(getChainCode(a[0], a[1], b[0], b[1]))
    return chainCode
 
 
# '''This function generates the list of points for 
# a straight line using Bresenham's Algorithm'''
def Bresenham2D(x1, y1, x2, y2):
    ListOfPoints = []
    ListOfPoints.append([x1, y1])
    xdif = x2 - x1
    ydif = y2 - y1
    dx = abs(xdif)
    dy = abs(ydif)
    if(xdif > 0):
        xs = 1
    else:
        xs = -1
    if (ydif > 0):
        ys = 1
    else:
        ys = -1
    if (dx > dy):
 
        # Driving axis is the X-axis
        p = 2 * dy - dx
        while (x1 != x2):
            x1 += xs
            if (p >= 0):
                y1 += ys
                p -= 2 * dx
            p += 2 * dy
            ListOfPoints.append([x1, y1])
    else:
 
        # Driving axis is the Y-axis
        p = 2 * dx-dy
        while(y1 != y2):
            y1 += ys
            if (p >= 0):
                x1 += xs
                p -= 2 * dy
            p += 2 * dx
            ListOfPoints.append([x1, y1])
    return ListOfPoints
 
def DriverFunction(image):
     # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    st.image([binary.astype(np.uint8)], caption=["Binary Image"], use_column_width=True)
    # st.write(type(binary))
    # a = list(chain.from_iterable(binary))

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    koordinat = list(chain.from_iterable(contours))
    st.write(koordinat)
    (x1, y1) = koordinat[0][0][0],koordinat[0][0][1]
    (x2, y2) = koordinat[1][0][0],koordinat[1][0][1]
    (x3, y3) = koordinat[2][0][0],koordinat[2][0][1]
    (x4, y4) = koordinat[3][0][0],koordinat[3][0][1]
    
    ListOfPoints = Bresenham2D(x1, y1, x2, y2)
    ListOfPoints2 = Bresenham2D(x2, y2, x3, y3)
    ListOfPoints3 = Bresenham2D(x3, y3, x4, y4)
    ListOfPoints4 = Bresenham2D(x4, y4, x1, y1)
    chainCode = generateChainCode(ListOfPoints)
    chainCode2 = generateChainCode(ListOfPoints2)
    chainCode3 = generateChainCode(ListOfPoints3)
    chainCode4 = generateChainCode(ListOfPoints4)
    chainCodeString = "".join(str(e) for e in chainCode)
    chainCodeString2 = "".join(str(e) for e in chainCode2)
    chainCodeString3 = "".join(str(e) for e in chainCode3)
    chainCodeString4 = "".join(str(e) for e in chainCode4)
    st.write('Chain code is', chainCodeString,chainCodeString2,chainCodeString3,chainCodeString4)
 


# Function to compute chain code
def compute_chain_code(image, direction):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    # st.write(type(binary))
    # a = list(chain.from_iterable(binary))

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    st.image([binary.astype(np.uint8)], caption=["Binary Image"], use_column_width=True)

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    st.write(list(chain.from_iterable(largest_contour)))
    chain_code_4dir =[]
    chain_code_8dir=[]
    # Compute the chain code for the largest contour
    if direction == 4 :
      chain_code_4dir = compute_chain_code_for_contour(largest_contour, directions=4)
    elif direction == 8:
      chain_code_8dir = compute_chain_code_for_contour(largest_contour, directions=8)

    return chain_code_4dir, chain_code_8dir, largest_contour


# Function to compute chain code for a contour
def compute_chain_code_for_contour(contour, directions=4):
    chain_code = []
    # Define the possible directions based on the number of directions
    if directions == 4:
        possible_directions = [0, 1, 2, 3]
    elif directions == 8:
        possible_directions = [0, 1, 2, 3, 4, 5, 6, 7]
    else:
        raise ValueError("Invalid number of directions. Use 4 or 8.")

    # Define the mapping of directions for 4 and 8 directions
    dir_mapping_4dir = {0: 0, 1: 1, 2: 2, 3: 3}
    dir_mapping_8dir = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}

    # Iterate through the points in the contour to compute the chain code
    for i in range(len(contour) - 1):
        start_point = contour[i][0]
        end_point = contour[i + 1][0]

        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]

        # Compute the direction based on arctan(dy / dx)
        angle = np.degrees(np.arctan2(dy, dx))

        # Quantize the angle to one of the possible directions
        if directions == 4:
            direction = min(possible_directions, key=lambda x: abs(angle - x * 90))
            chain_code.append(dir_mapping_4dir[direction])
        elif directions == 8:
            direction = min(possible_directions, key=lambda x: abs(angle - x * 45))
            chain_code.append(dir_mapping_8dir[direction])

    return chain_code



def Restoration(image,option):
    if option == "8 Direction":
      # DriverFunction(image)
      _, chain_code_8dir, _ = compute_chain_code(image,8)
      st.subheader("Chain Code (8-direction):")
      st.write(chain_code_8dir)
    elif option == "4 Direction":
      chain_code_4dir, _, _ = compute_chain_code(image,4)
      st.subheader("Chain Code (4-direction):")
      st.write(chain_code_4dir)
      # st.image(image, caption="Original Image", use_column_width=True)
      
      #  # Convert the image to grayscale
      # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      
      # # Apply edge detection (you can use a more sophisticated method based on your needs)
      # edges = cv2.Canny(gray, 50, 150)
      
      # # Find contours in the image
      # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      # st.image(edges, caption="edges Image", use_column_width=True)
      # st.write("contours")
      # st.write(contours)
      
      # # Chain code calculation (just a simple example here)
      # chain_code_result = []
      # for contour in contours:
      #     for point in contour:
      #         x, y = point[0]
      #         chain_code_result.append((x, y))
      
      # st.write("Chain Code Result:")
      # st.write(chain_code_result)
      # st.header("Freeman Code 3, 3, 3, 6, 6, 4, 6, 7, 7, 0, 0, 6")
      # freeman_code = [3, 3, 3, 6, 6, 4, 6, 7, 7, 0, 0, 6] 

      # img = np.zeros((10,10))

      # x, y = 4, 4 
      # img[y][x] = 1
      # for direction in freeman_code:
      #     if direction in [1,2,3]:
      #         y -= 1
      #     if direction in [5,6,7]:
      #         y += 1
      #     if direction in  [3,4,5]:
      #         x -= 1
      #     if direction in [0,1,7]:
      #         x += 1

      #     img[y][x] = 1
      # st.image(img, width=200)

      # image=np.array(image)
      # if len(image.shape) == 3:
      #     image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

      #  # Threshold the image to create a binary image
      # _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
      # # Find the contours in the binary image
      # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
      # st.write(thresh)
      # st.write(contours)

      # # Compute the chain code for each contour
      # for contour in contours:
      #     epsilon = 0.02 * cv2.arcLength(contour, True)
      #     approx = cv2.approxPolyDP(contour, epsilon, True)
      #     st.write("spprox")
      #     st.write(approx)
      #     chain_code = []
      #     for i in range(len(approx)):
      #         x1, y1 = approx[i][0]
      #         x2, y2 = approx[(i + 1) % len(approx)][0]  # Wrap around to the first point
      #         dx = x2 - x1
      #         dy = y2 - y1
      #         if dx == 0 and dy == 1:
      #             code = 0
      #         elif dx == -1 and dy == 0:
      #             code = 1
      #         elif dx == 0 and dy == -1:
      #             code = 2
      #         elif dx == 1 and dy == 0:
      #             code = 3
      #         else:
      #             code = -1  # Handle invalid input
      #         chain_code.append(code)
      #     st.write("Chain Code")
      #     st.write(chain_code)

    # elif option == "8 Direction":
    #   image=np.array(image)
    #   if len(image.shape) == 3:
    #       image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #    # Threshold the image to create a binary image
    #   _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    #   # Find the contours in the binary image
    #   contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #   # Compute the chain code for each contour
    #   for contour in contours:
    #     chain = cv2.approxPolyDP(contour, 1, True)
    #     chain_code = []
    #     for i in range(len(chain)):
    #         x1, y1 = chain[i][0]
    #         x2, y2 = chain[(i + 1) % len(chain)][0]  # Wrap around to the first point
    #         dx = x2 - x1
    #         dy = y2 - y1
    #         if dx == 0 and dy == 1:
    #             code = 0
    #         elif dx == -1 and dy == 1:
    #             code = 1
    #         elif dx == -1 and dy == 0:
    #             code = 2
    #         elif dx == -1 and dy == -1:
    #             code = 3
    #         elif dx == 0 and dy == -1:
    #             code = 4
    #         elif dx == 1 and dy == -1:
    #             code = 5
    #         elif dx == 1 and dy == 0:
    #             code = 6
    #         elif dx == 1 and dy == 1:
    #             code = 7
    #         else:
    #             code = -1  # Handle invalid input
    #         chain_code.append(code)
    #   # st.image([image, thresh.astype(np .uint8)], caption=["Original Image", "Binary Image",], use_column_width=True)
def get_frequency(data):
    frequency = {}
    for byte in data:
        frequency[byte] = frequency.get(byte, 0) + 1
    return frequency

def build_huffman_tree(frequency):
    heap = [[weight, [byte, ""]] for byte, weight in frequency.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    
    return heap[0][1:]

def compress_image_huffman(image):
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format='PNG')
    img_data = img_byte_array.getvalue()

    # Get frequency of each byte in the image data
    frequency = get_frequency(img_data)

    # Build Huffman tree
    huffman_tree = build_huffman_tree(frequency)

    # Create a dictionary for encoding
    encoding_dict = {byte: code for byte, code in huffman_tree}

    # Encode image using Huffman encoding
    encoded_data = ''.join(encoding_dict[byte] for byte in img_data)

    return encoded_data, encoding_dict

def decompress_image(encoded_data, encoding_dict):
    # Decode image using Huffman encoding
    decoded_data = huffman.decode(encoding_dict, encoded_data)
    return Image.frombytes('RGB', (1, len(decoded_data)), bytes(decoded_data))


def compress_image_img_comp(original_image, reduction_factor=64):
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
def compress(uncompressed):
    """Compress a string to a list of output symbols."""

    # Build the dictionary.
    dict_size = 256
    dictionary = dict((chr(i), i) for i in range(dict_size))
    # in Python 3: dictionary = {chr(i): i for i in range(dict_size)}

    w = ""
    result = []
    for c in uncompressed:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            # Add wc to the dictionary.
            dictionary[wc] = dict_size
            dict_size += 1
            w = c

    # Output the code for w.
    if w:
        result.append(dictionary[w])
    return result


def decompress(compressed):
    """Decompress a list of output ks to a string."""
    from io import StringIO
    # st.write(compressed)

    # Build the dictionary.
    dict_size = 256
    dictionary = dict((i, chr(i)) for i in range(dict_size))
    # in Python 3: dictionary = {i: chr(i) for i in range(dict_size)}

    # use StringIO, otherwise this becomes O(N^2)
    # due to string concatenation in a loop
    result = StringIO()
    w = chr(compressed.pop(0))
    result.write(w)
    for k in compressed:
        if k in dictionary:
            entry = dictionary[k]
        elif k == dict_size:
            entry = w + w[0]
        else:
            raise ValueError('Bad compressed k: %s' % k)
        result.write(entry)

        # Add w+entry[0] to the dictionary.
        dictionary[dict_size] = w + entry[0]
        dict_size += 1

        w = entry
    return result.getvalue()



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
        option = st.selectbox('Pilih Filter',('Pilih','Morfologi','Geometri', 'Restoration','Huffman','Compression','LZW'))
        st.write('Kamu memilih:',option)

#---------Choose from selectbox------------

        if option == 'Select':
            pass

#--------Morfologi in selectionbox---
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

        #--------Geometri in selectionbox---
        elif option == 'Geometri':
            option2 = st.selectbox('Pilih Filter',('Pilih','Outlier','Rank Order','Low Pass Filter','Filter Median'))
            if option2 is not None:
                # st.image(image) 
                # result = Morfologi(image_cv2,option2)
                Geometri(image_cv2,option2)
        
        elif option == 'Restoration':
            option2= st.selectbox('Pilih Filter',('Pilih','8 Direction','4 Direction'))
            if option2 is not None:
                # st.image(image) 
                # result = Morfologi(image_cv2,option2)
                Restoration(image_cv2,option2) 

        elif option == 'Huffman':
            st.title("Huffman Coding Image Compression App")

            operation = st.radio("Select Operation", ["Compress Image", "Decompress Image"])

            if operation == "Compress Image":
                # uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

                # if uploaded_file is not None:
                # original_image = Image.open(uploaded_file)

                st.image(image, caption="Original Image", use_column_width=True)

                    # Compress the image
                encoded_data, encoding_dict = compress_image_huffman(image)

                    # Provide a download link for the compressed data
                st.markdown("### Download Compressed Data")
                st.markdown(
                        f'<a href="data:application/octet-stream;base64,{base64.b64encode(encoded_data.encode()).decode()}" download="compressed_data.bin">Download Compressed Data</a>',
                        unsafe_allow_html=True
                    )
            else:
                uploaded_file = st.file_uploader("Upload compressed data...", type="bin")

                if uploaded_file is not None:
                    # Read the uploaded compressed data
                    compressed_data = uploaded_file.read().decode()

                    # Decode the compressed data
                    decoded_image = decompress_image(compressed_data, encoding_dict)

                    st.image(decoded_image, caption="Decompressed Image", use_column_width=True)


        elif option == 'Compression':
            st.title("Lossy Image Compression App")

            # uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

            if image is not None:
                # original_image = Image.open(uploaded_file)

                st.image(image, caption="Original Image", use_column_width=True)

                reduction_factor = st.slider("Color Reduction Factor", min_value=2, max_value=256, value=64)

                # Compress the image using the lossy compression method
                compressed_data = compress_image_img_comp(image, reduction_factor=reduction_factor)

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
        
        elif option == 'LZW':
            # operation = st.radio("Select Operation", ["Compress", "Decompress"])
            # How to use:
            input = st.text_input("Enter your text")
            st.title("Compressed")  
            compressed = compress(input)
            # if operation == "Compress":
            #input text
            st.markdown(compressed)
            # elif operation == "Decompress":
            # input = st.text_input("Enter your code [seperate with comma]")
            # input = input.split(", ")
            st.title("Decompressed")   
            decompressed = decompress(compressed)
            st.markdown(decompressed)   
                  
            

        else:
            pass

if __name__ =="__main__":
    main()

