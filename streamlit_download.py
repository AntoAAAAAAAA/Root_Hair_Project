import streamlit as st

st.title("Discovering Signals - Root Hair Analyzer")
st.divider()

st.text("Please select a download option from below")

col1, col2 = st.columns(2)

with col1.expander("MacOS", expanded=False):
    # st.text("Click the button below, and then select 'Download' within the Google Drive folder")
    st.link_button('Download App', "https://drive.google.com/file/d/1KbLhJGJvmK7tSITIEX0bBLjokr3tuENp/view?usp=drive_link")

with col2.expander("Windows", expanded=False):
    st.text("A Windows version of the app is currently in development. " \
    "Please check back in the future once development is complete. Thank you!  ")       

st.text(" ")

st.text("After clicking 'download', make sure to find the .zip file in your files, double click it, and you " \
"should see an app named 'Root Hair Analyzer'. For convenience, feel free to move that app to your desktop " \
"for easier future access.")