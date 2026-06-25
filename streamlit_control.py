import streamlit as st

st.set_page_config(layout="wide")

instr = st.Page("streamlit_instr.py", title="Instructions")
main_page = st.Page("streamlit_main.py", title="Root Hair Analyzer")

pg = st.navigation([instr, main_page])
pg.run()