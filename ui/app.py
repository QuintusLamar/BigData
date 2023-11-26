import streamlit as st
from gtts import gTTS
from io import BytesIO
from PIL import Image
from models.BLIP import run_BLIP


def text_to_speech(text):
    audio_bytes = BytesIO()
    tts = gTTS(text=text, lang="en")
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes.read()


st.title("Our application")

model_output = ""
uploaded_file = st.file_uploader("Choose the image you would like described.")

if uploaded_file is not None:

    file_extension = uploaded_file.name.split('.')[-1].lower()

    if file_extension not in ['jpg', 'jpeg', 'png']:

        st.error("Please upload a valid JPG or PNG file.")

    else:

        image = Image.open(uploaded_file)

        model_output = ""
        st.image(image, use_column_width=True)

        if st.button("Run on model"):

            placeholder = st.empty()

            with st.spinner("Running ..."):

                model_output = run_BLIP(image)
                placeholder.success("Done...")

            if model_output != "":
                st.write(model_output)
