import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import scipy as scipy
import plotly as plot
import streamlit as st
from main import *

if 'conversion factor' not in st.session_state:
    st.session_state['conversion factor'] = 0.0000000000

st.title('Root Hair Analyzer')

### -----Microscope convertion factor--------
microscope_conversion_factor = st.number_input(
    'Microscope Conversion Factor:', 
    min_value=0.0, 
    step=0.00000001, 
    format='%.10f',
    value= st.session_state['conversion factor'])

st.session_state['conversion factor'] = microscope_conversion_factor

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["bmp", "png", "jpg", "jpeg", "tif", "tiff"],
    key='1'
)
uploaded_file.seek(0) # reset file pointer to the beginnign for next read 

image_gray = None
if uploaded_file is not None:
    # file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    st.session_state['image1'] = image_gray

    st.image(image_gray, caption="Grayscale image", )

st.text('')
st.text('')
st.text('')
st.text('')
st.text('')
st.text('')
st.text('')

if st.button("Analyze", key='analysis1'):
    fig1 = main(image_gray, microscope_conversion_factor)
    fig1.update_layout(autosize = False,
                    width = 1000,
                    height = 800)
    st.session_state['fig1'] = fig1

    st.plotly_chart(fig1, use_container_width=True)


'''
User inputs a conversion factor that is saved to session state and stays that way 

grayscale image is uploaded, then saved to session state
If: the user uploads a NEW image (need to check if its new)
- the session state updates
- session state is presented in st.image()
else:
- the current session state is presented 

When user hits analyze
If: there is an image in session state
- analyze that image using main()
- save the resulting fig to session state
- if there is a present plotly figure
    - display it 
else:
- error message that says 'no file uploaded' or something like that 





'''