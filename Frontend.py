import streamlit as st
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from Backend import *

#--------------implement Screen-------------
#--------Pertama Upload Gambar-----------------
#-----------Convert Gambar ke array----------
#---------Kedua Pilih Filter from Kotak-----

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

        image_cv2_2 = image_cv2
        image_cv2 = cv2.cvtColor(image_cv2,cv2.COLOR_BGR2BGRA)
        # option = st.selectbox('Pilih Filter',('Pilih','Edge Detection','Grayscale','Negative Transformation','Gaussian Blur','Reduce Noise','Sharping'))
        option = st.selectbox('Pilih Filter',('Pilih','Resize','Edge Detection','Blurred','Histogram','Segmentasi', 'Beauty', 'Binarized Image', "Pixel Sorting"))
        st.write('Kamu memilih:',option)

#---------Choose from selectbox------------

        if option == 'Select':
            pass

#--------Edge Detection in selectionbox---
        elif option == 'Edge Detection':
            st.header('Gambar yang diinput')
            st.image(image)
            t1 =  st.select_slider("Ukuran threshold 1",range(0,256), value =100)
            t2 =  st.select_slider("Ukuran threshold 2",range(0,256), value =100)
            st.markdown('Gambar setelah Edge Detection (Canny)')
            # st.image(edge_detection(image_cv2),t1,t2)
            st.image(edge_detection(image_cv2,t1,t2))


#--------Blurred in selectionbox----------------
        elif option =='Blurred':
            option2 = st.selectbox('Pilih Filter',('Pilih','Mean Blur','Gaussian Blur','Median Blur'))
            if option2 is not None:
                input =  st.select_slider("ukuran kernel",range(3,100,2))
                st.image(image)
                result = Blurred(image_cv2,input,option2)
                if result is not None:
                    st.markdown('Gambar setelah '+ option2)
                    st.image(result)

#--------Histogram in selectionbox----------------
        elif option =='Histogram':
            st.header('Gambar yang diinput')
            st.image(image)
            st.markdown('Gambar setelah '+ option)
            image_eq = equalizeHistogram(image_cv2)
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
            ax = axes.ravel()

            ax[0].imshow(image, cmap=plt.cm.gray)
            ax[0].set_title('Original')

            ax[1].imshow(image_eq, cmap=plt.cm.gray)
            ax[1].set_title('Equalized')

            for a in ax:
                a.axis('off')

            st.pyplot(fig)

            #tampilkan histogram
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
            ax = axes.ravel()

            ax[0].hist(image_cv2.ravel(), bins=256, histtype='step', color='black')
            ax[0].set_title('Original Histogram')

            ax[1].hist(image_eq.ravel(), bins=256, histtype='step', color='black')
            ax[1].set_title('Equalized Histogram')

            st.pyplot(fig)
            
            
            # st.image(edge_detection(image_cv2),t1,t2)
            # st.pyplot(cv2.calcHist([image_cv2],[0],None,[256],[0,256]))
            # st.image(cv2.equalizeHist(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY))

#--------Segmentasi in Selectionbox----------------
        #secara defaul deteksi kuning atau hijau
        elif option =='Segmentasi':
            option2 = st.selectbox('Pilih Filter',('Pilih','Bitwise','Thresholding','Morfologi'))
            if option2 is not None:
                st.image(image)
                
                result = Segmentasi(image_cv2,option2)
                # result = Blurred(image_cv2,input,option2)
                if result is not None:
                    st.markdown('Gambar setelah '+ option2)
                    st.image(result, clamp=True)


            

# --------Resizein selectionbox-----
        elif option =='Resize':
            panjang =st.select_slider("panjang",range(0,500), value = 200)
            lebar = st.select_slider("lebar",range(0,500), value = 200)

            # option2 = st.selectbox('Pilih Ukuran',('Pilih','','interpolasi'))
            st.header('Gambar yang diinput')
            st.image(image)
            result = Resize(image_cv2,panjang, lebar)
            # if result is not None:
            st.markdown('Gambar setelah Resize')
            st.image(result)
                    

# --------Beauty Mode in selectionbox-----
        elif option =='Beauty':
            # option2 = st.selectbox('Pilih Ukuran',('Pilih','','interpolasi'))
            st.header('Gambar yang diinput')
            st.image(image)
            result = Beauty(image_cv2_2)
            # if result is not None:
            st.markdown('Hasil Beauty Mode')
            st.image(result)

# --------BinarizeImage in selectionbox-----
        elif option == 'Binarized Image':
            st.header('Gambar yang diinput')
            st.image(image)
            image = gray_scale(image_cv2)
            # Set the threshold value (adjust as needed)
            threshold_value = 128

            # Binarize the image
            binary_image = binarize_image(image, threshold_value)
            # result = binarize_image(image_cv2)
            # if result is not None:
            st.markdown('Gambar setelah Binary Image')
            st.image(binary_image)
                    

# --------pixel Sorting  in selectionbox-----
        elif option =='Pixel Sorting':
            
            # option2 = st.selectbox('Pilih Ukuran',('Pilih','','interpolasi'))
            st.header('Gambar yang diinput')
            st.image(image)
            result = pixelSorting(image_cv2)
            #melakukan sort image
            sort_all_pixels(image)
            st.markdown('Hasil sortall')
            resultimage = Image.open('output-sortall.png')
            st.image(resultimage, clamp=True, channels='RGB')
            # if result is not None:
            st.markdown('Hasil Sorting Baris')
            pixel_sort_baris(image, 50)
            resultimage = Image.open('output-sortIntervalsRow.png')
            st.image(resultimage)
            st.markdown('Hasil Sorting Kolom')
            # pixel_sort_baris(image, 50)
            pixel_sort_kolom(image, 30)
            resultimage = Image.open('output-sortIntervalsColumn.png')
            st.image(resultimage)

            # st.markdown('Hasil Sorting')






















# --------Gray Scale in selectionbox-----
        # elif option == 'Grayscale':
        #     st.header('Gambar yang diinput')
        #     st.image(image)
        #     st.markdown('Gambar setelah Grayscale')
        #     st.image(gray_scale(image_cv2))

#--------Reduce Noise in selectionbox----------------

        # elif option =='Reduce Noise (Mean)':
        #     st.header('Input image')
        #     st.image(image)
        #     st.markdown('Image after Remove Noise')
        #     st.image(reduce_noise(image_cv2))

#------Negative Transformationin selectionbox----------
#         elif option == 'Negative Transformation':
#             st.header('Input image')
#             st.image(image)
#             st.markdown('Image after Negative Transformation')
#             st.image(negative_transformation(image_cv2))

# #------Gaussian Blur in selectionbox-------------------

#         elif option =='Gaussian Blur':
#             st.header('Input image')
#             st.image(image)
#             st.markdown('Image after Gaussian Blurring')
#             st.image(Gaussian_Blur(image_cv2))



#-----------Sharping in selectionbox----------------

        # elif option =='Sharping':
        #     st.header('Input image')
        #     st.image(image)
        #     st.markdown('Image after Sharping')
        #     st.image(sharp_image(image_cv2))

        else:
            pass

if __name__ =="__main__":
    main()