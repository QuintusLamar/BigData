import streamlit as st
from PIL import Image

st.title("Our application")

uploaded_file = st.file_uploader("Choose the image you would like described.")

if uploaded_file is not None:

    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension not in ['jpg', 'jpeg', 'png']:
        st.error("Please upload a valid JPG or PNG file.")
    else:
        image = Image.open(uploaded_file)

        st.image(image, caption="Uploaded Image.", use_column_width=True)

        st.write("Image Metadata:")
        st.write(f"File Name: {uploaded_file.name}")
        st.write(f"Image Size: {image.size[0]}x{image.size[1]} pixels")
