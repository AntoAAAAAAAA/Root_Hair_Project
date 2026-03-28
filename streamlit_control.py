import streamlit as st

main_page = st.Page("streamlit_main.py", title="Root Hair Analyzer")
# page_2 = st.Page("page_2.py", title="Background")

pg = st.navigation([main_page])
pg.run()