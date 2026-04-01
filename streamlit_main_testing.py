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
if 'image1' not in st.session_state:
    st.session_state['image1'] = None
if 'fig1' not in st.session_state:
    st.session_state['fig1'] = None
if 'image2' not in st.session_state:
    st.session_state['image2'] = None
if 'fig2' not in st.session_state:
    st.session_state['fig2'] = None

st.title('Root Hair Analyzer')

### -----Microscope convertion factor--------
microscope_conversion_factor = st.number_input(
    'Microscope Conversion Factor:', 
    min_value=0.0, 
    step=0.00000001, 
    format='%.10f',
    value= st.session_state['conversion factor'])

st.session_state['conversion factor'] = microscope_conversion_factor

### -----Length parameters--------

column1, column2 = st.columns(2)
with column1:
    upper = st.number_input(
        'Upper length threshold (in microns):', 
        min_value=0.0, 
        step=0.1, 
        format='%.1f',
        value= 100.0)

with column2:
    lower = st.number_input(
        'Lower length threshold (in microns):', 
        min_value=0.0, 
        step=0.1, 
        format='%.1f',
        value= 30.0)

## -----Image upload and analysis per column--------
col1, col2 = st.columns(2)

with col1:
    uploaded_file1 = st.file_uploader(
        "Upload an image",
        type=["bmp", "png", "jpg", "jpeg", "tif", "tiff"],
        key='1'
    )

    image_gray1 = None
    if uploaded_file1 is not None:
        uploaded_file1.seek(0) # reset file pointer to the beginning  
        # file_bytes2 = np.asarray(bytearray(uploaded_file2.getvalue()), dtype=np.uint8)
        file_bytes1 = np.frombuffer(uploaded_file1.read(), np.uint8)
        image_gray1 = cv2.imdecode(file_bytes1, cv2.IMREAD_GRAYSCALE)
        st.session_state['image1'] = image_gray1

        st.image(image_gray1, caption="Grayscale image", )

    st.text('')
    st.text('')
    st.text('')
    st.text('')
    st.text('')
    st.text('')
    st.text('')

    if st.button("Analyze", key='analysis1'):
        imageToAnalyze = st.session_state['image1']
        fig1 = main(imageToAnalyze, microscope_conversion_factor, upper, lower)
        fig1.update_layout(autosize = False,
                        width = 1000,
                        height = 800)
        st.session_state['fig1'] = fig1

    if st.session_state['fig1'] is not None:
        st.plotly_chart(st.session_state['fig1'], use_container_width=True)

with col2:
    uploaded_file2 = st.file_uploader(
        "Upload an image",
        type=["bmp", "png", "jpg", "jpeg", "tif", "tiff"],
        key='2'
    )
          
    image_gray2 = None
    if uploaded_file2 is not None:
        uploaded_file2.seek(0) # reset file pointer to the beginning
        # file_bytes2 = np.asarray(bytearray(uploaded_file2.getvalue()), dtype=np.uint8)
        file_bytes2 = np.frombuffer(uploaded_file2.read(), np.uint8)
        image_gray2 = cv2.imdecode(file_bytes2, cv2.IMREAD_GRAYSCALE)
        st.session_state['image2'] = image_gray2

        st.image(image_gray2, caption="Grayscale image", )

    st.text('')
    st.text('')
    st.text('')
    st.text('')
    st.text('')
    st.text('')
    st.text('')

    if st.button("Analyze", key='analysis2'):
        imageToAnalyz2 = st.session_state['image2']
        fig2 = main(imageToAnalyz2, microscope_conversion_factor, upper, lower)
        fig2.update_layout(autosize = False,
                        width = 1000,
                        height = 800)
        st.session_state['fig2'] = fig2

    if st.session_state['fig2'] is not None:
        st.plotly_chart(st.session_state['fig2'], use_container_width=True)