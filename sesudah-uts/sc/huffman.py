import streamlit as st
from PIL import Image
import io
import heapq
import huffman
import base64

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

def compress_image(image):
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

def main():
    st.title("Huffman Coding Image Compression App")

    operation = st.radio("Select Operation", ["Compress Image", "Decompress Image"])

    if operation == "Compress Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            original_image = Image.open(uploaded_file)

            st.image(original_image, caption="Original Image", use_column_width=True)

            # Compress the image
            encoded_data, encoding_dict = compress_image(original_image)

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

if __name__ == "__main__":
    main()
