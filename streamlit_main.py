import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import scipy as scipy
import plotly as plot
import streamlit as st
import plotly.express as px
from src.main_v2 import *

if 'conversion factor' not in st.session_state:
    st.session_state['conversion factor'] = 0.0

if 'image_gray1' not in st.session_state:
    st.session_state['image_gray1'] = None
if 'image_color1' not in st.session_state:
    st.session_state['image_color1'] = None
if 'fig1' not in st.session_state:
    st.session_state['fig1'] = None
if 'traces1' not in st.session_state:
    st.session_state['traces1'] = None
if 'hair_index1' not in st.session_state:
    st.session_state['hair_index1'] = 0
if 'image_list1' not in st.session_state:
    st.session_state['image_list1'] = None
if 'selected_image1' not in st.session_state:
    st.session_state['selected_image1'] = None


if 'image_gray2' not in st.session_state:
    st.session_state['image_gray2'] = None
if 'image_color2' not in st.session_state:
    st.session_state['image_color2'] = None
if 'fig2' not in st.session_state:
    st.session_state['fig2'] = None
if 'traces2' not in st.session_state:
    st.session_state['traces2'] = None
if 'hair_index2' not in st.session_state:
    st.session_state['hair_index2'] = 0
if 'image_list2' not in st.session_state:
    st.session_state['image_list2'] = None
if 'selected_image2' not in st.session_state:
    st.session_state['selected_image2'] = None


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
        st.session_state['image_gray1'] = image_gray1
        image_color1 = cv2.imdecode(file_bytes1, cv2.IMREAD_COLOR)
        st.session_state['image_color1'] = image_color1
        st.image(image_gray1, caption="Grayscale image", )

    co1, co2, co3 = st.columns(3)
    with co2: 
        if st.button("Analyze", key='analysis1'):
            if st.session_state['image_gray1'] is not None:
                image_gray1 = st.session_state['image_gray1']
                image = st.session_state['image_color1']
                fig1, traces1 = main_v2(image,image_gray1, microscope_conversion_factor, upper, lower)
                st.session_state['traces1'] = traces1
                image_list1 = []
                image_list1.insert(0, fig1)
                for trace in traces1:

                    base_image = px.imshow(image_gray1, binary_string=True)
                    
                    base_image.data[0].hovertemplate = None
                    base_image.data[0].hoverinfo = "skip"

                    base_image.add_trace(trace)
                    base_image.update_yaxes(visible=False)
                    base_image.update_xaxes(visible=False)

                    image_list1.append(base_image)

                # fig1.update_layout(autosize = False,
                #                 width = 1000,
                #                 height = 800)
                st.session_state["image_list1"] = image_list1
                st.session_state["hair_index1"] = 0
                st.session_state['selected_image1'] = image_list1[0]

            else:
                st.error("No image for analysis provided or in memory")

    if st.session_state['image_list1'] is not None:
        colu1, colu2, colu3 = st.columns(3)
        
        with colu1:
            st.text('')
            st.text('')
            st.text('')
            if st.button('Previous', key='previous_hair1'):
                    st.session_state['hair_index1'] -= 1

                    if st.session_state["hair_index1"] < 0:
                        st.session_state["hair_index1"] = len(st.session_state["image_list1"])

                    st.session_state['selected_image1'] = st.session_state.image_list1[st.session_state['hair_index1']]
            
        with colu3:
            st.text('')
            st.text('')
            st.text('')
            if st.button('Next', key='next_hair1'):
                    st.session_state['hair_index1'] += 1

                    if st.session_state["hair_index1"] >= len(st.session_state["image_list1"]):
                        st.session_state["hair_index1"] = 0

                    st.session_state['selected_image1'] = st.session_state.image_list1[st.session_state['hair_index1']]

        if st.session_state['selected_image1'] is not None:
            st.plotly_chart(st.session_state['selected_image1'], width='stretch')

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
        st.session_state['image_gray2'] = image_gray2
        image_color2 = cv2.imdecode(file_bytes2, cv2.IMREAD_COLOR)
        st.session_state['image_color2'] = image_color2

        st.image(image_gray2, caption="Grayscale image", )

    st.text('')
    st.text('')
    st.text('')
    st.text('')
    st.text('')
    st.text('')
    st.text('')

    if st.button("Analyze", key='analysis2'):
        if st.session_state['image_gray2'] is not None:
            imageToAnalyze2 = st.session_state['image_gray2']
            image = st.session_state['image_color2']
            fig2, traces2 = main_v2(image, imageToAnalyze2, microscope_conversion_factor, upper, lower)
            fig2.update_layout(autosize = False,
                            width = 1000,
                            height = 800)
            st.session_state['fig2'] = fig2
            st.session_state['traces2'] = traces2
        else:
            st.error("No image for analysis provided or in memory")

    if st.session_state['fig2'] is not None:
        st.plotly_chart(st.session_state['fig2'], width='stretch')


