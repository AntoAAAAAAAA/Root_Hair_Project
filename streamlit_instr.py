import streamlit as st

st.header("Root Hair Analyzer", divider="rainbow")

st.write(
    "This tool was developed for use in **Dr. Clark’s Freshman Research Initiative lab** "
    "to help students measure root hairs from microscopy images. It was created by an "
    "undergraduate mentor as a **supplementary analysis tool** and is intended to support "
    "existing measurement workflows (ex. ImageJ). Because root images vary in quality and complexity, " \
    "all measurements should be reviewed manually before being used for analysis."
)


st.info(
    "**Important:** This program is best used on relatively clean root images with clearly visible individual root hairs. " \
    "Messy or overlapping hairs may produce inaccurate detections, " \
    "so manual verification is strongly recommended."
)

st.write(
    "For questions, issues, or feedback, contact: "
    "**[anto2005antony@gmail.com](mailto:anto2005antony@gmail.com)**"
)

st.text(" ")
st.text(" ")
st.text(" ")
st.text(" ")
c1, c2, c3 = st.columns(3)
with c2:
    if st.button("Go to Root Hair Analyzer", type="primary", use_container_width=True):
        st.switch_page("streamlit_main.py")
st.text(" ")
st.text(" ")
st.text(" ")
st.text(" ")

st.header("How to Use the Program", divider="rainbow")

st.write(
    """#### 1) Enter analysis settings
- Enter the **microscope conversion factor** for the microscope used to capture the images.
- Set the **upper** and **lower** length thresholds for valid root hairs.
- Recommended starting values: **110 µm** for the upper threshold and **10 µm** for the lower threshold.
"""
)

st.write(
    """#### 2) Upload images
- Upload the **T0 image** in the **left column**.
- Upload the **T1 image** in the **right column**.
"""
)

st.write(
    """#### 3) Run the analysis
- Click **Analyze** on each side.
- After analysis, an interactive image will appear below each upload panel showing the root hairs detected by the program.
"""
)

st.write(
    """#### 4) Review individual root hairs
- Use the **Previous** and **Next** buttons to cycle through detected root hairs one at a time.
- Hover over a highlighted root hair to view its measured length.
- The currently selected root hair length will also be displayed beneath the image.
"""
)

st.write(
    """#### 5) Add measurements to the table
- If a highlighted root hair appears correct, click **Add to Table** to add that measurement to the T0 or T1 column below.
- When possible, add matching **T0** and **T1** measurements to the same row so they remain paired for later analysis.
"""
)

st.write(
    """#### 6) Export / transfer measurements
- The table at the bottom of the page can be copied into an Excel worksheet for downstream analysis.
"""
)

st.text(" ")
st.text(" ")


st.error(
    "This tool is intended to assist with measurement, not to serve as a final source of truth. "
    "Always review highlighted hairs before recording results."
)