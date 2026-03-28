import streamlit as st
import cv2
from main import *

st.title('Root Hair Analyzer')

microscope_conversion_factor = st.number_input(
    'Microscope Conversion Factor:', 0.0, step=0.00000001, format='%.10f')


# file upload stuff 
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["bmp", "png", "jpg", "jpeg", "tif", "tiff"]
)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.write("Original filename:", uploaded_file.name)

    if image_gray is None:
        st.error("OpenCV could not read the uploaded image.")

    else:
        st.write("Image loaded successfully.")
        st.image(image_gray, caption="Grayscale image")

if st.button("Analyze"):
    fig = main(image_gray, microscope_conversion_factor)
    fig.update_layout(autosize = False,
                      width = 1000,
                      height = 800)
    st.plotly_chart(fig)


'''
Next steps: 
This now desperately needs integration with session_state

'''
