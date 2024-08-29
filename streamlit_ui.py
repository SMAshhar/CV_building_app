import streamlit as st


# title
st.title("CV Builder App built by SMAshhar")

# taking in inputs from users
jobLink = st.text_input("Paste Job posting text here.")
user_writeup= st.text_input("Paste write a brief writeup about yourself, your experience and education.")
github= st.text_input("Your gihub link please")
cv_data= st.text_area("Paste everything you have done so far here.")

# Submitting 
if st.button("Submit"):
 with st.spinner('Please wait while our agents are making the most suited CV for you. Might take some minutes'):
    from app import cv_builder
    result = cv_builder(jobLink, cv_data, user_writeup="", github="")
    st.success('Done!')
    st.write("Customized CV as per given data.")


