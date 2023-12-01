import streamlit as st
from gtts import gTTS
from io import BytesIO
from PIL import Image
from models.BLIP import get_caption
from models.BLIP import get_vqa


def text_to_speech(text):
    audio_bytes = BytesIO()
    tts = gTTS(text=text, lang="en")
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes.read()


st.title("Our application")

if "model_output" not in st.session_state.keys():
    st.session_state.model_output = ""

if "vqa_output" not in st.session_state.keys():
    st.session_state.vqa_output = ""

if "file_name" not in st.session_state.keys():
    st.session_state.file_name = ""

uploaded_file = st.file_uploader("Choose the image you would like described.")

model = st.selectbox("Select the model you would like to use.", ["BLIP", "GIT"])
model = model.lower()

model_type = st.selectbox(
    "Select the model size you would like to use.", ["Base", "Large"]
)
model_type = model_type.lower()

if uploaded_file is not None:
    if st.session_state.file_name != uploaded_file.name:
        st.session_state.vqa_output = ""
        st.session_state.model_output = ""
    st.session_state.file_name = uploaded_file.name

    file_extension = uploaded_file.name.split(".")[-1].lower()
    if file_extension not in ["jpg", "jpeg", "png"]:
        st.error("Please upload a valid JPG or PNG file.")
    else:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        if st.button("Run on model"):
            st.session_state.model_output = ""
            # placeholder = st.empty()
            with st.spinner("Running ..."):
                model_output = get_caption(image, model, model_type)
                st.session_state.model_output = model_output
                # placeholder.success("Done...")
        
        if st.session_state.model_output != "":
            st.text_area("Model Output" , st.session_state.model_output)
            st.audio(text_to_speech(st.session_state.model_output))
            question = st.text_input(label="Enter your question about the image")
            if st.button("Get answer"):
                if (len(question) == 0):
                    st.error("Please enter a question.")
                else:
                    # placeholder = st.empty()
                    with st.spinner("Running ..."):
                        st.session_state.vqa_output = ""
                        vqa_output = get_vqa(image, question, model, model_type)
                        st.session_state.vqa_output = vqa_output
                        # placeholder.success("Done...")
                    if st.session_state.vqa_output != "":
                        st.text_area("Question Answer", vqa_output)
                        st.audio(text_to_speech(vqa_output))
