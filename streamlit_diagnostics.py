### -----Imports-------
import os
import time

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import psutil
import streamlit as st
from ultralytics import SAM

from src.main_v2 import *

# TESTING
import os
import psutil

PROCESS = psutil.Process(os.getpid())


def logMemory(stage):
    rss_gb = PROCESS.memory_info().rss / (1024 ** 3)
    print(f"{stage}: {rss_gb:.3f} GB RSS", flush=True)


logMemory("Python process started")

import streamlit as st
logMemory("After importing Streamlit")

import numpy as np
logMemory("After importing NumPy")

import pandas as pd
logMemory("After importing pandas")

import cv2
logMemory("After importing OpenCV")

import scipy
logMemory("After importing SciPy")

import skimage
logMemory("After importing scikit-image")

import plotly.express as px
logMemory("After importing Plotly")

from ultralytics import SAM
logMemory("After importing Ultralytics SAM")

from src.main_v2 import *
logMemory("After importing main_v2")
# TESTING


### -----Memory diagnostic helpers--------

PROCESS = psutil.Process(os.getpid())


def getMemoryGB():
    """Return the current process resident-set size in gigabytes."""
    return PROCESS.memory_info().rss / (1024 ** 3)


def logMemory(stage):
    """Print current process memory and return it."""
    current_memory_gb = getMemoryGB()

    print(
        f"[MEMORY] {stage}: "
        f"{current_memory_gb:.3f} GB RSS",
        flush=True
    )

    return current_memory_gb


def logMemoryChange(stage, startingMemoryGB):
    """Print current process memory and its change from an earlier value."""
    current_memory_gb = getMemoryGB()
    difference_gb = current_memory_gb - startingMemoryGB

    print(
        f"[MEMORY] {stage}: "
        f"{current_memory_gb:.3f} GB RSS "
        f"({difference_gb:+.3f} GB)",
        flush=True
    )

    return current_memory_gb


def showMemory(
    debugContainer,
    stage,
    startingMemoryGB=None
):
    """
    Display current process memory in the Streamlit interface and
    print the same measurement to the terminal or deployment logs.
    """
    current_memory_gb = getMemoryGB()

    if startingMemoryGB is None:
        message = (
            f"**{stage}:** "
            f"`{current_memory_gb:.3f} GB RSS`"
        )

    else:
        difference_gb = (
            current_memory_gb
            - startingMemoryGB
        )

        message = (
            f"**{stage}:** "
            f"`{current_memory_gb:.3f} GB RSS` "
            f"(`{difference_gb:+.3f} GB`)"
        )

    debugContainer.write(message)

    print(
        f"[MEMORY] {stage}: "
        f"{current_memory_gb:.3f} GB"
        + (
            ""
            if startingMemoryGB is None
            else (
                f" "
                f"({current_memory_gb - startingMemoryGB:+.3f} GB)"
            )
        ),
        flush=True
    )

    return current_memory_gb


def logArrayMemory(name, array):
    """Print shape, data type, and raw NumPy storage size."""
    if array is None:
        print(
            f"[ARRAY] {name}: None",
            flush=True
        )

        return

    size_mb = (
        array.nbytes
        / (1024 ** 2)
    )

    print(
        f"[ARRAY] {name}: "
        f"shape={array.shape}, "
        f"dtype={array.dtype}, "
        f"size={size_mb:.2f} MB",
        flush=True
    )


def describeSessionValue(value):
    """Create a compact description of a session-state value."""
    if value is None:
        return "None"

    if isinstance(value, np.ndarray):
        size_mb = (
            value.nbytes
            / (1024 ** 2)
        )

        return (
            f"NumPy array, "
            f"shape={value.shape}, "
            f"dtype={value.dtype}, "
            f"{size_mb:.2f} MB"
        )

    if isinstance(value, list):
        return (
            f"list with "
            f"{len(value)} items"
        )

    if isinstance(value, dict):
        return (
            f"dictionary with "
            f"{len(value)} entries"
        )

    return type(value).__name__


def summarizeSessionState(label):
    """Print relevant large session-state entries."""
    tracked_keys = [
        "image_gray1",
        "image_color1",
        "traces1",
        "fig1",
        "image_list1",
        "selected_image1",
        "image_gray2",
        "image_color2",
        "traces2",
        "fig2",
        "image_list2",
        "selected_image2"
    ]

    print(
        f"\n[SESSION STATE] {label}",
        flush=True
    )

    for key in tracked_keys:
        value = st.session_state.get(key)

        print(
            f"    {key}: "
            f"{describeSessionValue(value)}",
            flush=True
        )

    logMemory(
        f"Session-state summary complete: "
        f"{label}"
    )


### -----Cached SAM model--------

@st.cache_resource(
    show_spinner=False,
    show_time=False
)
def loadSamModel():
    print(
        "\n[SAM] Cache miss: "
        "constructing sam2_l.pt",
        flush=True
    )

    before_model = logMemory(
        "Immediately before SAM construction"
    )

    model = SAM(
        "sam2_l.pt"
    )

    logMemoryChange(
        "Immediately after SAM construction",
        before_model
    )

    return model


logMemory(
    "Before loading cached SAM model"
)

model = loadSamModel()

logMemory(
    "After retrieving cached SAM model"
)


### -----Session State Initiations--------

if "conversion factor" not in st.session_state:
    st.session_state[
        "conversion factor"
    ] = 0.0


session_defaults = {
    "image_gray1": None,
    "image_color1": None,
    "fig1": None,
    "traces1": None,
    "hair_index1": 0,
    "image_list1": None,
    "selected_image1": None,
    "length1": None,
    "col1_list": [],

    "image_gray2": None,
    "image_color2": None,
    "fig2": None,
    "traces2": None,
    "hair_index2": 0,
    "image_list2": None,
    "selected_image2": None,
    "length2": None,
    "col2_list": []
}


for key, default_value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


if "final_table" not in st.session_state:
    st.session_state[
        "final_table"
    ] = pd.DataFrame(
        columns=[
            "T0 Measurements (μm)",
            "T1 Measurements (μm)"
        ]
    )


summarizeSessionState(
    "After session-state initialization"
)


### -----Title--------

st.title(
    "Root Hair Analyzer"
)

st.divider()


left, middle, right = st.columns(3)


if left.button(
    "Go back to Instructions",
    type="primary",
    use_container_width=True
):
    st.switch_page(
        "streamlit_instr.py"
    )


st.text("")
st.text("")
st.text("")
st.text("")


### -----Microscope conversion factor--------

column1, column2 = st.columns(2)


microscope_conversion_factor = (
    column1.number_input(
        "Microscope Conversion Factor:",
        min_value=0.0,
        step=0.00000001,
        format="%.10f",
        value=st.session_state[
            "conversion factor"
        ]
    )
)


st.text(" ")


st.session_state[
    "conversion factor"
] = microscope_conversion_factor


### -----Length parameters--------

column1, column2 = st.columns(2)


with column1:
    upper = st.number_input(
        "Upper length threshold "
        "(in microns):",
        min_value=0.0,
        step=0.1,
        format="%.1f",
        value=100.0
    )


with column2:
    lower = st.number_input(
        "Lower length threshold "
        "(in microns):",
        min_value=0.0,
        step=0.1,
        format="%.1f",
        value=30.0
    )


### -----Image upload and analysis per column--------

col1, col2 = st.columns(2)

def selectImage(image_gray, fig_input, hairidx, traces: list):
    fig = px.imshow(image_gray, binary_string=True)

    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.data[0].hovertemplate = None
    fig.data[0].hoverinfo = "skip"

    if hairidx == 0:
        fig = fig_input
    else:
        selected_trace = traces[hairidx - 1][0]
        fig.add_trace(selected_trace)

    fig.update_yaxes(visible=False)
    fig.update_xaxes(visible=False)

    return fig 
def column(
    uploadKey,
    imageGrayKey,
    imageColorKey,
    imageCaption,
    analyzeButtonKey,
    conversionFactorKey,
    tracesKey,
    figKey,
    imageListKey,
    hairIdxKey,
    selectedImagesKey,
    previousArrowKey,
    forwardArrowKey,
    plotlyKey,
    colListKey,
    addToTableKey,
    T
):

    ### -----Visible memory diagnostics--------

    with st.expander(
        f"{T} Memory Diagnostics",
        expanded=True
    ):
        debugContainer = st.container()

        debugContainer.caption(
            "Memory shown is the resident memory used by the full "
            "Python process, including NumPy, OpenCV, PyTorch, "
            "SAM, Streamlit, and Plotly."
        )

        showMemory(
            debugContainer,
            f"{T}: Beginning of Streamlit rerun"
        )


    ### -----Upload image--------

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=[
            "bmp",
            "png",
            "jpg",
            "jpeg",
            "tif",
            "tiff"
        ],
        key=uploadKey
    )


    if uploaded_file is not None:
        upload_memory_start = showMemory(
            debugContainer,
            f"{T}: Before reading uploaded file"
        )

        uploaded_file.seek(0)

        raw_bytes = uploaded_file.read()

        file_bytes = np.frombuffer(
            raw_bytes,
            dtype=np.uint8
        )

        image_gray = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_GRAYSCALE
        )

        image_color = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )


        if image_gray is None or image_color is None:
            st.error(
                "The uploaded image could not be decoded."
            )

            return


        st.session_state[
            imageGrayKey
        ] = image_gray

        st.session_state[
            imageColorKey
        ] = image_color


        debugContainer.write(
            f"**{T} uploaded file:** "
            f"`{file_bytes.nbytes / (1024 ** 2):.2f} MB compressed`"
        )

        debugContainer.write(
            f"**{T} grayscale image:** "
            f"`shape={image_gray.shape}`, "
            f"`dtype={image_gray.dtype}`, "
            f"`{image_gray.nbytes / (1024 ** 2):.2f} MB`"
        )

        debugContainer.write(
            f"**{T} color image:** "
            f"`shape={image_color.shape}`, "
            f"`dtype={image_color.dtype}`, "
            f"`{image_color.nbytes / (1024 ** 2):.2f} MB`"
        )


        logArrayMemory(
            f"{T} grayscale image",
            image_gray
        )

        logArrayMemory(
            f"{T} color image",
            image_color
        )


        showMemory(
            debugContainer,
            f"{T}: After decoding uploaded image",
            upload_memory_start
        )


        st.image(
            uploaded_file,
            caption=imageCaption
        )


    ### -----Analyze button--------

    co1, co2, co3, co4, co5, co6, co7 = (
        st.columns(7)
    )


    with co4:
        st.text(" ")

        if st.button(
            "Analyze",
            key=analyzeButtonKey,
            type="primary"
        ):
            debugContainer.empty()

            debugContainer.markdown(
                f"### {T} Analysis Run"
            )

            debugContainer.write(
                f"**Length thresholds:** "
                f"`lower={lower:.1f} µm`, "
                f"`upper={upper:.1f} µm`"
            )


            if (
                st.session_state[
                    conversionFactorKey
                ]
                == 0.0
            ):
                st.toast(
                    "Microscope conversion factor is missing!"
                )


            elif (
                st.session_state[
                    imageGrayKey
                ]
                is None
            ):
                st.toast(
                    "No image for analysis provided "
                    "or in memory!"
                )


            else:
                analysis_memory_start = showMemory(
                    debugContainer,
                    f"{T}: Before analysis"
                )


                with st.spinner(
                    "",
                    show_time=True
                ):
                    print(
                        "\n"
                        + "#" * 70,
                        flush=True
                    )

                    print(
                        f"STREAMLIT ANALYSIS STARTED: {T}",
                        flush=True
                    )

                    print(
                        f"Upper threshold: {upper}",
                        flush=True
                    )

                    print(
                        f"Lower threshold: {lower}",
                        flush=True
                    )

                    print(
                        "#" * 70,
                        flush=True
                    )


                    summarizeSessionState(
                        f"{T}: Immediately before main_v2"
                    )


                    image_gray = (
                        st.session_state[
                            imageGrayKey
                        ]
                    )

                    image_color = (
                        st.session_state[
                            imageColorKey
                        ]
                    )


                    main_v2_memory_start = showMemory(
                        debugContainer,
                        f"{T}: Immediately before main_v2"
                    )


                    main_v2_time_start = (
                        time.perf_counter()
                    )


                    results = main_v2(
                        image_color,
                        image_gray,
                        microscope_conversion_factor,
                        upper,
                        lower,
                        model
                    )


                    main_v2_elapsed = (
                        time.perf_counter()
                        - main_v2_time_start
                    )


                    showMemory(
                        debugContainer,
                        f"{T}: Immediately after main_v2",
                        main_v2_memory_start
                    )


                    debugContainer.write(
                        f"**`main_v2` runtime:** "
                        f"`{main_v2_elapsed:.2f} seconds`"
                    )


                    fig = results[
                        "fig"
                    ]

                    traces = results[
                        "traces"
                    ]

                    valid_root_hair_masks = (
                        results.get(
                            "valid_root_hair_masks",
                            []
                        )
                    )


                    debugContainer.write(
                        f"**Plotly traces returned:** "
                        f"`{len(traces)}`"
                    )

                    debugContainer.write(
                        f"**Valid root hairs returned:** "
                        f"`{len(valid_root_hair_masks)}`"
                    )


                    ### -----Mask-storage diagnostics--------

                    if (
                        len(valid_root_hair_masks)
                        > 0
                    ):
                        total_mask_bytes = 0
                        total_thicker_mask_bytes = 0


                        for hair in valid_root_hair_masks:
                            hair_mask = hair.get(
                                "mask",
                                hair.get(
                                    "root hair mask"
                                )
                            )

                            thicker_mask = hair.get(
                                "thicker mask",
                                hair.get(
                                    "thicker_mask"
                                )
                            )


                            if (
                                isinstance(
                                    hair_mask,
                                    np.ndarray
                                )
                            ):
                                total_mask_bytes += (
                                    hair_mask.nbytes
                                )


                            if (
                                isinstance(
                                    thicker_mask,
                                    np.ndarray
                                )
                            ):
                                total_thicker_mask_bytes += (
                                    thicker_mask.nbytes
                                )


                        debugContainer.write(
                            f"**Stored full-size hair masks:** "
                            f"`{total_mask_bytes / (1024 ** 2):.2f} MB`"
                        )

                        debugContainer.write(
                            f"**Stored thicker masks:** "
                            f"`{total_thicker_mask_bytes / (1024 ** 2):.2f} MB`"
                        )


                        print(
                            f"[MASK STORAGE] {T}: "
                            f"full masks="
                            f"{total_mask_bytes / (1024 ** 2):.2f} MB, "
                            f"thicker masks="
                            f"{total_thicker_mask_bytes / (1024 ** 2):.2f} MB",
                            flush=True
                        )


                    ### -----Store fig and traces--------

                    before_session_assignment = (
                        showMemory(
                            debugContainer,
                            f"{T}: Before storing fig and traces "
                            f"in session state"
                        )
                    )


                    st.session_state[
                        tracesKey
                    ] = traces

                    st.session_state[
                        figKey
                    ] = fig


                    showMemory(
                        debugContainer,
                        f"{T}: After storing fig and traces "
                        f"in session state",
                        before_session_assignment
                    )

                    ### -----Store image list--------

                    before_selected_figure_assignment = (
                        showMemory(
                            debugContainer,
                            f"{T}: Before storing selected figure "
                            f"in session state"
                        )
                    )
                    st.session_state[
                        hairIdxKey
                    ] = 0

                    st.session_state[
                        selectedImagesKey
                    ] = fig


                    showMemory(
                        debugContainer,
                        f"{T}: After storing image_list "
                        f"in session state",
                        before_selected_figure_assignment
                    )


                    summarizeSessionState(
                        f"{T}: After complete analysis"
                    )


                    print(
                        f"STREAMLIT ANALYSIS FINISHED: {T}",
                        flush=True
                    )

                    print(
                        "#" * 70
                        + "\n",
                        flush=True
                    )


                showMemory(
                    debugContainer,
                    f"{T}: Spinner block completed",
                    analysis_memory_start
                )


    ### -----Figure navigation--------

    if (
        st.session_state[
            tracesKey
        ]
        is not None
    ):
        st.divider()


        colu1, colu2, colu3, colu4, colu5 = (
            st.columns(5)
        )


        with colu1:
                    if st.button('←', key= previousArrowKey):
                        st.session_state[hairIdxKey] -= 1
        
                        if st.session_state[hairIdxKey] < 0:
                            st.session_state[hairIdxKey] = len(st.session_state[tracesKey])
        
                        st.session_state[selectedImagesKey] = selectImage(
                            image_gray = st.session_state[imageGrayKey],
                            fig_input = st.session_state[figKey],
                            hairidx= st.session_state[hairIdxKey],
                            traces= st.session_state[tracesKey]
                            )


        with colu5:
                    if st.button('→', key= forwardArrowKey):
                        st.session_state[hairIdxKey] += 1
        
                        if st.session_state[hairIdxKey] > len(st.session_state[tracesKey]):
                            st.session_state[hairIdxKey] = 0
        
                        st.session_state[selectedImagesKey] = selectImage(
                            image_gray = st.session_state[imageGrayKey],
                            fig_input = st.session_state[figKey],
                            hairidx= st.session_state[hairIdxKey],
                            traces= st.session_state[tracesKey]
                            )

        if (
            st.session_state[
                selectedImagesKey
            ]
            is not None
        ):
            render_memory_start = (
                showMemory(
                    debugContainer,
                    f"{T}: Immediately before "
                    f"st.plotly_chart"
                )
            )


            render_time_start = (
                time.perf_counter()
            )


            st.plotly_chart(
                st.session_state[
                    selectedImagesKey
                ],
                width="stretch",
                height="stretch",
                key=plotlyKey
            )


            render_elapsed = (
                time.perf_counter()
                - render_time_start
            )


            showMemory(
                debugContainer,
                f"{T}: Immediately after "
                f"st.plotly_chart",
                render_memory_start
            )


            debugContainer.write(
                f"**`st.plotly_chart` Python call time:** "
                f"`{render_elapsed:.2f} seconds`"
            )


            indx = st.session_state[
                hairIdxKey
            ]


            if indx == 0:
                colol1, colol2, colol3 = (
                    st.columns(
                        3,
                        vertical_alignment="center"
                    )
                )

                colol2.text(
                    "Displaying all valid "
                    "root hairs found"
                )


            else:
                length = (
                    st.session_state[
                        tracesKey
                    ][
                        indx - 1
                    ][
                        1
                    ]
                )


                colu3.text(
                    f"Length: "
                    f"{length:.2f} µm"
                )


                if colu3.button(
                    "Add to Table",
                    key=addToTableKey
                ):
                    st.session_state[
                        colListKey
                    ].append(
                        length
                    )

                    st.toast(
                        f"{length:.2f} "
                        f"added to {T}",
                        icon="➕"
                    )


### -----Create T0 column--------

with col1:
    column(
        uploadKey="1",
        imageGrayKey="image_gray1",
        imageColorKey="image_color1",
        imageCaption="T0 image",
        analyzeButtonKey="analysis1",
        conversionFactorKey="conversion factor",
        tracesKey="traces1",
        figKey="fig1",
        imageListKey="image_list1",
        hairIdxKey="hair_index1",
        selectedImagesKey="selected_image1",
        previousArrowKey="previous_hair1",
        forwardArrowKey="next_hair1",
        plotlyKey="plotly1",
        colListKey="col1_list",
        addToTableKey="add_to_table1",
        T="T0"
    )


### -----Create T1 column--------

with col2:
    column(
        uploadKey="2",
        imageGrayKey="image_gray2",
        imageColorKey="image_color2",
        imageCaption="T1 image",
        analyzeButtonKey="analysis2",
        conversionFactorKey="conversion factor",
        tracesKey="traces2",
        figKey="fig2",
        imageListKey="image_list2",
        hairIdxKey="hair_index2",
        selectedImagesKey="selected_image2",
        previousArrowKey="previous_hair2",
        forwardArrowKey="next_hair2",
        plotlyKey="plotly2",
        colListKey="col2_list",
        addToTableKey="add_to_table2",
        T="T1"
    )


st.divider()


### -----Measurements Table--------

t0 = st.session_state[
    "col1_list"
]

t1 = st.session_state[
    "col2_list"
]


max_len = max(
    len(t0),
    len(t1),
    1
)


t0_padded = (
    t0
    + [None]
    * (
        max_len
        - len(t0)
    )
)


t1_padded = (
    t1
    + [None]
    * (
        max_len
        - len(t1)
    )
)


table_df = pd.DataFrame(
    {
        "T0 Measurements (μm)": (
            t0_padded
        ),

        "T1 Measurements (μm)": (
            t1_padded
        )
    }
)


edited_table = st.data_editor(
    table_df,
    height=250,
    hide_index=True,
    num_rows="dynamic"
)


st.session_state[
    "col1_list"
] = (
    pd.to_numeric(
        edited_table[
            "T0 Measurements (μm)"
        ],
        errors="coerce"
    )
    .dropna()
    .tolist()
)


st.session_state[
    "col2_list"
] = (
    pd.to_numeric(
        edited_table[
            "T1 Measurements (μm)"
        ],
        errors="coerce"
    )
    .dropna()
    .tolist()
)


### -----Remove Entries from Table--------

remove_col1, remove_col2 = st.columns(2)


with remove_col1:
    if (
        len(
            st.session_state[
                "col1_list"
            ]
        )
        > 0
    ):
        t0_index = st.selectbox(
            "Remove T0 value",
            options=list(
                range(
                    len(
                        st.session_state[
                            "col1_list"
                        ]
                    )
                )
            ),
            format_func=lambda i: (
                f"{i + 1}: "
                f"{st.session_state['col1_list'][i]:.2f} µm"
            ),
            key="remove_t0_select"
        )


        if st.button(
            "Remove selected T0",
            key="remove_t0_button"
        ):
            st.session_state[
                "col1_list"
            ].pop(
                t0_index
            )

            st.rerun()


with remove_col2:
    if (
        len(
            st.session_state[
                "col2_list"
            ]
        )
        > 0
    ):
        t1_index = st.selectbox(
            "Remove T1 value",
            options=list(
                range(
                    len(
                        st.session_state[
                            "col2_list"
                        ]
                    )
                )
            ),
            format_func=lambda i: (
                f"{i + 1}: "
                f"{st.session_state['col2_list'][i]:.2f} µm"
            ),
            key="remove_t1_select"
        )


        if st.button(
            "Remove selected T1",
            key="remove_t1_button"
        ):
            st.session_state[
                "col2_list"
            ].pop(
                t1_index
            )

            st.rerun()


logMemory(
    "End of Streamlit script rerun"
)