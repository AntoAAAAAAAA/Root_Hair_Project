import streamlit as st


instr = st.Page("streamlit_instr.py", title="Instructions", default=True)
main_page = st.Page("streamlit_main.py", title="Root Hair Analyzer")

pg = st.navigation([instr, main_page])

if pg == instr:
    current_layout = 'centered'
if pg == main_page:
    current_layout = 'wide'

st.set_page_config(layout=current_layout)

pg.run()

with st.sidebar:
    st.caption("Made in :streamlit: by Anto A.")
