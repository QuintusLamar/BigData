import streamlit as st
from gtts import gTTS
from io import BytesIO
from PIL import Image
from models.BLIP import get_caption
from models.BLIP import get_vqa

from models.LLaMA import LLaMA
from models.Mistral import Mistral

from pseudocode import chooseBestNQuestions
import datetime
import os
from loguru import logger

log_folder = "logs"
log_file = f"{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.log"
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
logger.add(os.path.join(log_folder, log_file), enqueue=True)

replicate_api_key = "r8_TNoIA5wIRHm9v4jUgwkcpNaSqkHURB23PhkK8"
llama = LLaMA(replicate_api_key)
mistral = Mistral(replicate_api_key)


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

if "final_response" not in st.session_state.keys():
    st.session_state.final_response = ""

uploaded_file = st.file_uploader("Choose the image you would like described.")

model = st.selectbox("Select the model you would like to use.", ["BLIP", "GIT"])
model = model.lower()

model_type = st.selectbox(
    "Select the model size you would like to use.", ["Base", "Large"]
)
model_type = model_type.lower()

llm_type = st.selectbox(
    "Select the model you would like to use for summarization.",
    ["LLaMA-2-70B", "Mistral-7B"],
)
llm_type = llm_type.lower()

if uploaded_file is not None:
    if st.session_state.file_name != uploaded_file.name:
        st.session_state.vqa_output = ""
        st.session_state.model_output = ""
        st.session_state.final_response = ""
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
                questions_list_1 = llama.get_questions(model_output, num_questions=5)
                questions_list_2 = mistral.get_questions(model_output, num_questions=5)
                questions_list = chooseBestNQuestions(
                    questions_list_1, questions_list_2, 5
                )

                qna_list = []
                for question in questions_list:
                    vqa_output = get_vqa(image, question, model, model_type)
                    qna_list.append(f"Question: {question}, Answer: {vqa_output}")
                logger.info(f"questions: {qna_list}")
                if llm_type == "mistral-7b":
                    response = mistral.get_complete_summary(qna_list)
                else:
                    response = llama.get_complete_summary(qna_list)
                response = "".join(response)
                logger.info(f"summary: {response}")
                st.session_state.final_response = response
                # placeholder.success(response)

        if st.session_state.final_response != "":
            st.text_area("Image summary: ", st.session_state.final_response)
            st.audio(text_to_speech(st.session_state.final_response))

        if st.session_state.model_output != "":
            st.text_area("Model Output", st.session_state.model_output)
            st.audio(text_to_speech(st.session_state.model_output))
            question = st.text_input(label="Enter your question about the image")
            if st.button("Get answer"):
                if len(question) == 0:
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
