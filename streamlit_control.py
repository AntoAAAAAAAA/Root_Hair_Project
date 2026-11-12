import streamlit as st

import os
import sys
import psutil
import torch

# Configure CPU inference on Windows computers without CUDA
if sys.platform == "win32" and not torch.cuda.is_available():

    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)

    cpu_threads = min(12, physical_cores or logical_cores or 4)

    # nnU-Net reads this setting during import
    os.environ["nnUNet_def_n_proc"] = str(cpu_threads)

    # Configure PyTorch CPU parallelism
    torch.set_num_threads(cpu_threads)

    # print(f"Configured CPU inference: {cpu_threads} threads")
    
instr = st.Page("streamlit_instr.py", title="Instructions")
main_page = st.Page("streamlit_main.py", title="Root Hair Analyzer")

pg = st.navigation([instr, main_page])

if pg == instr:
    current_layout = 'centered'
if pg == main_page:
    current_layout = 'wide'
st.set_page_config(layout=current_layout)

pg.run()

# with st.sidebar:
#     st.caption("Made in :streamlit: by Anto A.")

with st.sidebar.container(key="sidebar_bottom"):
    st.write("Made in :streamlit: by Anto A.")
st.html("""
<style>
    .st-key-sidebar_bottom {
        position: absolute;
        bottom: 10px;
    }
</style>
""")
