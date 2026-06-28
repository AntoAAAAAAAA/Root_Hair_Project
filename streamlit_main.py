### -----Imports-------
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage as sk
import scipy as scipy
import plotly as plot
import streamlit as st
import plotly.express as px
from ultralytics import SAM
from src.main_v2 import *

### -----Session State Initiations--------

@st.cache_resource(show_spinner=False, show_time=False)
def loadSamModel():
    return SAM("sam2_l.pt")
model = loadSamModel()

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
if 'length1' not in st.session_state:
    st.session_state['length1'] = None
if 'col1_list' not in st.session_state:
    st.session_state['col1_list'] = []


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
if 'length2' not in st.session_state:
    st.session_state['length2'] = None
if 'col2_list' not in st.session_state:
    st.session_state['col2_list'] = []


if 'final_table' not in st.session_state:
    st.session_state['final_table'] = pd.DataFrame(columns=['T0 Measurements (μm)', 'T0 Measurements (μm)'])


### -----Title--------

st.title('Root Hair Analyzer')
st.divider()

left, middle, right = st.columns(3)
if left.button("Go back to Instructions", type="primary", use_container_width=True):
    st.switch_page("streamlit_instr.py")
st.text('')
st.text('')
st.text('')
st.text('')

### -----Microscope convertion factor--------
column1, column2 = st.columns(2)
microscope_conversion_factor = column1.number_input(    
    'Microscope Conversion Factor:', 
    min_value=0.0, 
    step=0.00000001, 
    format='%.10f',
    value= st.session_state['conversion factor'])
st.text(' ')

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
        file_bytes1 = np.frombuffer(uploaded_file1.read(), np.uint8)
        image_gray1 = cv2.imdecode(file_bytes1, cv2.IMREAD_GRAYSCALE)
        st.session_state['image_gray1'] = image_gray1
        image_color1 = cv2.imdecode(file_bytes1, cv2.IMREAD_COLOR)
        st.session_state['image_color1'] = image_color1
        st.image(uploaded_file1, caption="T0 image", )

    co1, co2, co3, co4, co5, co6, co7 = st.columns(7)
    with co4: 
        st.text(' ')
        if st.button("Analyze", key='analysis1', type='primary'):
           
            if st.session_state['conversion factor'] == 0.0:
                col1.error("Microscope conversion factor is missing")

            elif st.session_state['image_gray1'] is None:
                col1.error("No image for analysis provided or in memory")
            
            else:
                with st.spinner('', show_time=True):
                    image_gray1 = st.session_state['image_gray1']
                    image_color1 = st.session_state['image_color1']

                    fig1, traces1 = main_v2(image_color1, image_gray1, microscope_conversion_factor, upper, lower, model)
                    st.session_state['traces1'] = traces1
                    st.session_state['fig1'] = fig1

                    image_list1 = []
                    image_list1.insert(0, fig1)

                    for trace, length in traces1:
                        base_image = px.imshow(image_gray1, binary_string=True)
                        base_image.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                        
                        base_image.data[0].hovertemplate = None
                        base_image.data[0].hoverinfo = "skip"

                        base_image.add_trace(trace)
                        base_image.update_yaxes(visible=False)
                        base_image.update_xaxes(visible=False)

                        image_list1.append(base_image)

                    st.session_state["image_list1"] = image_list1
                    st.session_state["hair_index1"] = 0
                    st.session_state['selected_image1'] = image_list1[0]

            

    if st.session_state['image_list1'] is not None:
        st.divider()
        colu1, colu2, colu3, colu4, colu5 = st.columns(5)
        
        with colu1:
            if st.button('Previous', key='previous_hair1'):
                st.session_state['hair_index1'] -= 1

                if st.session_state["hair_index1"] < 0:
                    st.session_state["hair_index1"] = len(st.session_state["image_list1"]) - 1

                st.session_state['selected_image1'] = st.session_state.image_list1[st.session_state['hair_index1']]
            
        with colu5:
            if st.button('Next', key='next_hair1'):
                st.session_state['hair_index1'] += 1

                if st.session_state["hair_index1"] >= len(st.session_state["image_list1"]):
                    st.session_state["hair_index1"] = 0

                st.session_state['selected_image1'] = st.session_state.image_list1[st.session_state['hair_index1']]

        if st.session_state['selected_image1'] is not None:
            
            st.plotly_chart(st.session_state['selected_image1'], width='stretch', height='stretch', key='plotly1')
            
            indx = st.session_state.hair_index1
            if indx == 0:
                st.text("Displaying all valid root hairs found")
            else:
                length1 = st.session_state.traces1[indx-1][1]
                colu3.text(f'Length: {length1:.2f} µm')
                if colu3.button("Add to Table"):
                    st.session_state['col1_list'].append(length1)
                    st.toast(f'{length1:.2f} added to T0', icon='➕')

with col2:
    uploaded_file2 = st.file_uploader(
        "Upload an image",
        type=["bmp", "png", "jpg", "jpeg", "tif", "tiff"],
        key='2'
    )

    image_gray2 = None
    if uploaded_file2 is not None:
        uploaded_file2.seek(0) # reset file pointer to the beginning  
        file_bytes2 = np.frombuffer(uploaded_file2.read(), np.uint8)
        image_gray2 = cv2.imdecode(file_bytes2, cv2.IMREAD_GRAYSCALE)
        st.session_state['image_gray2'] = image_gray2
        image_color2 = cv2.imdecode(file_bytes2, cv2.IMREAD_COLOR)
        st.session_state['image_color2'] = image_color2
        st.image(uploaded_file2, caption="T1 image", )

    co1, co2, co3, co4, co5, co6, co7 = st.columns(7)
    with co4:  
        st.text(' ')
        if st.button("Analyze", key='analysis2', type='primary'):
            
            if st.session_state['conversion factor'] == 0.0:
                col2.error("Microscope conversion factor is missing")
            
            elif st.session_state['image_gray2'] is None:
                col2.error("No image for analysis provided or in memory")

            else:
                with st.spinner('', show_time=True):
                    image_gray2 = st.session_state['image_gray2']
                    image_color2 = st.session_state['image_color2']

                    fig2, traces2 = main_v2(image_color2, image_gray2, microscope_conversion_factor, upper, lower, model)
                    st.session_state['traces2'] = traces2
                    st.session_state['fig2'] = fig2

                    image_list2 = []
                    image_list2.insert(0, fig2)

                    for trace, length in traces2:
                        base_image = px.imshow(image_gray2, binary_string=True)
                        base_image.update_layout(margin=dict(l=0, r=0, t=0, b=0)    )

                        base_image.data[0].hovertemplate = None
                        base_image.data[0].hoverinfo = "skip"

                        base_image.add_trace(trace)
                        base_image.update_yaxes(visible=False)
                        base_image.update_xaxes(visible=False)

                        image_list2.append(base_image)

                    st.session_state["image_list2"] = image_list2
                    st.session_state["hair_index2"] = 0
                    st.session_state['selected_image2'] = image_list2[0]

           

    if st.session_state['image_list2'] is not None:
        st.divider()
        colu1, colu2, colu3, colu4, colu5 = st.columns(5)
        
        with colu1:
            if st.button('Previous', key='previous_hair2'):
                st.session_state['hair_index2'] -= 1

                if st.session_state["hair_index2"] < 0:
                    st.session_state["hair_index2"] = len(st.session_state["image_list2"]) - 1

                st.session_state['selected_image2'] = st.session_state.image_list2[st.session_state['hair_index2']]
            
        with colu5:
            if st.button('Next', key='next_hair2'):
                st.session_state['hair_index2'] += 1

                if st.session_state["hair_index2"] >= len(st.session_state["image_list2"]):
                    st.session_state["hair_index2"] = 0

                st.session_state['selected_image2'] = st.session_state.image_list2[st.session_state['hair_index2']]

        if st.session_state['selected_image2'] is not None:
            
            st.plotly_chart(st.session_state['selected_image2'], width='stretch', height='stretch', key='plotly2')
            
            indx = st.session_state.hair_index2
            if indx == 0:
                st.text("Displaying all valid root hairs found")
            else:
                length2 = st.session_state.traces2[indx-1][1]
                colu3.text(f'Length: {length2:.2f} µm')
                if colu3.button("Add to Table", key="add_to_table2"):
                    st.session_state['col2_list'].append(length2)
                    st.toast(f'{length2:.2f} added to T1', icon='➕')
st.divider()

### -----Measurements Table--------

t0 = st.session_state['col1_list']
t1 = st.session_state['col2_list']

max_len = max(len(t0), len(t1), 1)

t0_padded = t0 + [None] * (max_len - len(t0))
t1_padded = t1 + [None] * (max_len - len(t1))

table_df = pd.DataFrame({
            "T0 Measurements (μm)": t0_padded,
            "T1 Measurements (μm)": t1_padded
            })

edited_table = st.data_editor(table_df, height=250, hide_index=True, num_rows='dynamic')

st.session_state['col1_list'] = (pd.to_numeric(edited_table["T0 Measurements (μm)"], errors="coerce").dropna().tolist())
st.session_state['col2_list'] = (pd.to_numeric(edited_table["T1 Measurements (μm)"], errors="coerce").dropna().tolist())

### -----Remove Entries from Table--------

remove_col1, remove_col2 = st.columns(2)

with remove_col1:
    if len(st.session_state['col1_list']) > 0:
        t0_index = st.selectbox(
            "Remove T0 value",
            options=list(range(len(st.session_state['col1_list']))),
            format_func=lambda i: f"{i + 1}: {st.session_state['col1_list'][i]:.2f} µm",
            key="remove_t0_select"
        )

        if st.button("Remove selected T0", key="remove_t0_button"):
            st.session_state['col1_list'].pop(t0_index)
            st.rerun()

with remove_col2:
    if len(st.session_state['col2_list']) > 0:
        t1_index = st.selectbox(
            "Remove T1 value",
            options=list(range(len(st.session_state['col2_list']))),
            format_func=lambda i: f"{i + 1}: {st.session_state['col2_list'][i]:.2f} µm",
            key="remove_t1_select"
        )

        if st.button("Remove selected T1", key="remove_t1_button"):
            st.session_state['col2_list'].pop(t1_index)
            st.rerun()