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
import json
from platformdirs import user_data_dir


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
if 'selected_image1' not in st.session_state:
    st.session_state['selected_image1'] = None
if 'length1' not in st.session_state:
    st.session_state['length1'] = None


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
if 'selected_image2' not in st.session_state:
    st.session_state['selected_image2'] = None
if 'length2' not in st.session_state:
    st.session_state['length2'] = None


app_data_dir = Path(
    user_data_dir(
        appname="Root Hair Analyzer",
        appauthor=False
    ))
app_data_dir.mkdir(parents=True, exist_ok=True)
data_file = app_data_dir / "data.json"
if not data_file.exists():
    default_data = {
        "col1": [],
        "col2": []
    }
    with open(data_file, "w", encoding="utf-8") as file:
        json.dump(default_data, file, indent=2)

with open(data_file, 'r') as file:
    col_dict = json.load(file)
if 'col1_list' not in st.session_state:
    st.session_state['col1_list'] = col_dict['col1']
if 'col2_list' not in st.session_state:
    st.session_state['col2_list'] = col_dict['col2']


if 'final_table' not in st.session_state:
    st.session_state['final_table'] = pd.DataFrame(columns=['T0 Measurements (μm)', 'T1 Measurements (μm)'])


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

def selectImage(image_gray, fig_input, display_idx, traces: list):
    fig = px.imshow(image_gray, binary_string=True)

    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.data[0].hovertemplate = None
    fig.data[0].hoverinfo = "skip"

    if display_idx == 0:
        fig = fig_input
    else:
        trace_idx = display_idx - 1
        selected_trace = traces[trace_idx][0]
        fig.add_trace(selected_trace)

    fig.update_yaxes(visible=False)
    fig.update_xaxes(visible=False)

    return fig 

def column(uploadKey, imageGrayKey, imageColorKey, imageCaption, analyzeButtonKey, conversionFactorKey, tracesKey,
           figKey, imageListKey, displayIdxKey, selectedImagesKey, previousArrowKey, 
           forwardArrowKey, plotlyKey, colListKey, addToTableKey, T):
    
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["bmp", "png", "jpg", "jpeg", "tif", "tiff"],
        key=uploadKey
    )

    image_gray = None
    if uploaded_file is not None:
        uploaded_file.seek(0) # reset file pointer to the beginning  
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        st.session_state[imageGrayKey] = image_gray
        image_color = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.session_state[imageColorKey] = image_color
        st.image(uploaded_file, caption=imageCaption)

    co1, co2, co3, co4, co5, co6, co7 = st.columns(7)
    with co4: 
        st.text(' ')
        if st.button("Analyze", key=analyzeButtonKey, type='primary'):
           
            if st.session_state[conversionFactorKey] == 0.0:
                st.toast("Microscope conversion factor is missing!")

            elif st.session_state[imageGrayKey] is None:
                st.toast("No image for analysis provided or in memory!")
            
            else:
                with st.spinner('', show_time=True):
                    image_gray = st.session_state[imageGrayKey]
                    image_color = st.session_state[imageColorKey]

                    results = main_v2(image_color, image_gray, microscope_conversion_factor, upper, lower, model)
                    fig, traces = results['fig'], results['traces']
                    st.session_state[tracesKey] = traces
                    st.session_state[figKey] = fig
                    

                    # for trace, length in traces:
                    #     base_image = px.imshow(image_gray, binary_string=True)
                    #     base_image.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                        
                    #     base_image.data[0].hovertemplate = None
                    #     base_image.data[0].hoverinfo = "skip"

                    #     base_image.add_trace(trace)
                    #     base_image.update_yaxes(visible=False)
                    #     base_image.update_xaxes(visible=False)

                    #     image_list.append(base_image)

                    st.session_state[tracesKey] = traces
                    st.session_state[displayIdxKey] = 0
                    st.session_state[selectedImagesKey] = fig
            

    if st.session_state[tracesKey] is not None:
        st.divider()
        colu1, colu2, colu3 = st.columns([1,3,1], gap='xxlarge')
        
        with colu1:
            if st.button('←', key= previousArrowKey, width='stretch'):
                st.session_state[displayIdxKey] -= 1

                if st.session_state[displayIdxKey] < 0:
                    st.session_state[displayIdxKey] = len(st.session_state[tracesKey])

                st.session_state[selectedImagesKey] = selectImage(
                    image_gray = st.session_state[imageGrayKey],
                    fig_input = st.session_state[figKey],
                    display_idx= st.session_state[displayIdxKey],
                    traces= st.session_state[tracesKey]
                    )
                
        with colu3:
            if st.button('→', key= forwardArrowKey, width='stretch'):
                st.session_state[displayIdxKey] += 1

                if st.session_state[displayIdxKey] > len(st.session_state[tracesKey]):
                    st.session_state[displayIdxKey] = 0

                st.session_state[selectedImagesKey] = selectImage(
                    image_gray = st.session_state[imageGrayKey],
                    fig_input = st.session_state[figKey],
                    display_idx= st.session_state[displayIdxKey],
                    traces= st.session_state[tracesKey]
                    )

        if st.session_state[selectedImagesKey] is not None:
            
            st.plotly_chart(st.session_state[selectedImagesKey], width='stretch', height='stretch', key=plotlyKey)
            
            indx = st.session_state[displayIdxKey]
            if indx == 0:
                colol1, colol2, colol3 = st.columns(3, vertical_alignment='center')
                colol2.text("Displaying all valid root hairs found")
            else:
                length = st.session_state[tracesKey][indx-1][1]
                colu2.text(f'Length: {length:.2f} µm', text_alignment='center', width='stretch')
                if colu2.button("Add to Table", key = addToTableKey, width='stretch'):
                    st.session_state[colListKey].append(length)
                    st.toast(f'{length:.2f} added to {T}', icon='➕')

with col1:
    uploadKey = '1'
    imageGrayKey = 'image_gray1'
    imageColorKey = 'image_color1'
    imageCaption = 'T0 image'
    analyzeButtonKey = 'analysis1'
    conversionFactorKey = 'conversion factor'
    tracesKey = 'traces1'
    figKey = 'fig1'
    imageListKey = 'image_list1'
    displayIdxKey = 'hair_index1'
    selectedImagesKey = 'selected_image1'
    previousArrowKey = 'previous_hair1'
    forwardArrowKey = 'next_hair1'
    addToTableKey = 'add_to_table1'
    plotlyKey = 'plotly1'
    colListKey = 'col1_list'
    T = 'T0'
    column(uploadKey, imageGrayKey, imageColorKey, imageCaption, analyzeButtonKey, conversionFactorKey, tracesKey,
           figKey, imageListKey, displayIdxKey, selectedImagesKey, previousArrowKey, 
           forwardArrowKey, plotlyKey, colListKey, addToTableKey, T)


with col2:
    uploadKey = '2'
    imageGrayKey = 'image_gray2'
    imageColorKey = 'image_color2'
    imageCaption = 'T1 image'
    analyzeButtonKey = 'analysis2'
    conversionFactorKey = 'conversion factor'
    tracesKey = 'traces2'
    figKey = 'fig2'
    imageListKey = 'image_list2'
    displayIdxKey = 'hair_index2'
    selectedImagesKey = 'selected_image2'
    previousArrowKey = 'previous_hair2'
    forwardArrowKey = 'next_hair2'
    addToTableKey = 'add_to_table2'
    plotlyKey = 'plotly2'
    colListKey = 'col2_list'
    T = 'T1'
    column(uploadKey, imageGrayKey, imageColorKey, imageCaption, analyzeButtonKey, conversionFactorKey, tracesKey,
           figKey, imageListKey, displayIdxKey, selectedImagesKey, previousArrowKey, 
           forwardArrowKey, plotlyKey, colListKey, addToTableKey, T)

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


#### -----JSON File Updates--------
with open(data_file, 'r') as file:
    data = json.load(file)
data['col1'] = st.session_state['col1_list']
data['col2'] = st.session_state['col2_list']

with open(data_file, 'w') as file:
    json.dump(data, file, indent=2)


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