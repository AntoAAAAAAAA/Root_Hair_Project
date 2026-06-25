import streamlit as st

st.header("A Quick Intro:", divider='rainbow')
st.write(
    "This program is intended for use in Dr. Clark's Freshman Research Initiative lab. " \
    "It is intended to be used by student's to measure root hairs from root images. " \
    "This project was created by an undergraduate mentor in Dr. Clark's lab and is " \
    "not a professional tool intended for precision accuracy. It is a **supplemental tool** that is meant " \
    "to be used alongside student's current methods of measuring root hairs (ImageJ analysis). For any " \
    "question or concerns, please feel free to contact this email: anto2005antony@gmail.com")


st.text(' ')
st.text(' ')
st.header("How to Use this Program:", divider='rainbow')
st.write("1. Insert the microscope conversion factor from the specific microscope used for imaging")
st.write("2. Set your upper and lower limits for root hair length (It is recommended to set these as 110 and 10 respectively)")
st.write("3. Upload a T0 (left) and a T1 photo (right) of a root ")
st.write("4. Click 'Analyze' on both sides")
st.write("5. You will see an image populate below that shows highlights for the individual root hairs found by the program (note: This program will not work well with messy root hairs. It is not a professional tool and is far from perfect.)")
st.write("6. You can hover over blue, highlighted root hairs to find their length (The length is also displayed underneath the 'Analyze' buttons.)")
st.write("7. If you are happy with the root(s), click 'Add to Table', and the measurement(s) will be added to a column in the table below")
st.write("8. Ideally, you will find the same root hair on both images and add both measurements to the table as a pair")

st.text(' ')
st.write("Final note: For those in Dr. Clark's lab, this table can be directly copy-and-pasted into an excel template.")