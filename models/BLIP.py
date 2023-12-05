import os
import cv2
import numpy as np
import torch
import datetime
from collections import defaultdict

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
)
from transformers import AutoProcessor, AutoModelForCausalLM
from models.VQA import generateAnswer

# from loguru import logger

log_folder = "logs"
log_file = f"{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.log"
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
# logger.add(os.path.join(log_folder, log_file), enqueue=True)

import warnings

warnings.filterwarnings("ignore")


model_dict = defaultdict(dict)
processor_dict = defaultdict(dict)

# logger.info("Loading models...")
for model_type in ["base", "large"]:
    model_name = f"Salesforce/blip-image-captioning-{model_type}"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    processor_dict[f"blip_{model_type}"]["caption"] = processor
    model_dict[f"blip_{model_type}"]["caption"] = model

    if model_type == "base":
        model_name = "Salesforce/blip-vqa-base"
    else:
        model_name = "Salesforce/blip-vqa-capfilt-large"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForQuestionAnswering.from_pretrained(model_name)
    processor_dict[f"blip_{model_type}"]["vqa"] = processor
    model_dict[f"blip_{model_type}"]["vqa"] = model

for model_type in ["base", "large"]:
    model_name = f"microsoft/git-{model_type}-coco"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    processor_dict[f"git_{model_type}"]["caption"] = processor
    model_dict[f"git_{model_type}"]["caption"] = model

    model_name = f"microsoft/git-{model_type}-textvqa"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    processor_dict[f"git_{model_type}"]["vqa"] = processor
    model_dict[f"git_{model_type}"]["vqa"] = model
# logger.info("Loaded models...")


def get_blip_caption(
    image, model_type="base", text="a photography of", max_new_tokens=50
):
    # logger.info(f"{model_type}")
    processor = processor_dict[f"blip_{model_type}"]["caption"]
    model = model_dict[f"blip_{model_type}"]["caption"]
    inputs = processor(image, text, return_tensors="pt")
    caption_ids = model.generate(max_new_tokens=max_new_tokens, **inputs)
    response = processor.decode(caption_ids[0], skip_special_tokens=True)
    if (response is not None) and (len(response) > 0):
        # logger.info(f"response: {response}")
        return response
    else:
        # logger.warning(f"response: {response}")
        return "No response generated from the model."


def get_blip_vqa(image, question, model_type="base"):
    # logger.info(f"{model_type}")
    processor = processor_dict[f"blip_{model_type}"]["vqa"]
    model = model_dict[f"blip_{model_type}"]["vqa"]
    inputs = processor(image, question, return_tensors="pt")
    out = model.generate(**inputs)
    response = processor.decode(out[0], skip_special_tokens=True)
    if (response is not None) and (len(response) > 0):
        # logger.info(f"response: {response}")
        return response
    else:
        # logger.warning(f"response: {response}")
        return "No response generated from the model."


def get_git_caption(image, model_type="base"):
    # logger.info(f"{model_type}")
    processor = processor_dict[f"git_{model_type}"]["caption"]
    model = model_dict[f"git_{model_type}"]["caption"]
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    if (response is not None) and (len(response) > 0):
        # logger.info(f"response: {response}")
        return response
    else:
        # logger.warning(f"response: {response}")
        return "No response generated from the model."


def get_git_vqa(image, question, model_type="base"):
    # logger.info(f"{model_type}")
    processor = processor_dict[f"git_{model_type}"]["vqa"]
    model = model_dict[f"git_{model_type}"]["vqa"]
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    input_ids = processor(text=question, add_special_tokens=False).input_ids
    input_ids = [processor.tokenizer.cls_token_id] + input_ids
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    generated_ids = model.generate(
        pixel_values=pixel_values, input_ids=input_ids, max_length=50
    )
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)
    if (response is not None) and (len(response) > 0):
        # logger.info(f"response: {response}")
        return response[0].replace(question.lower(), "").strip()
    else:
        # logger.warning(f"response: {response}")
        return "No response generated from the model."


def get_caption(image, model="blip", model_type="base"):
    raw_image = np.array(image)
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    if model.lower() == "blip":
        caption = get_blip_caption(raw_image, model_type=model_type)
    else:
        caption = get_git_caption(raw_image, model_type=model_type)
    return caption


def get_vqa(image, question, model="blip", model_type="base"):
    # logger.info(f"Using {model}-{model_type}")
    raw_image = np.array(image)
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    if model.lower() == "blip":
        answer = get_blip_vqa(raw_image, question, model_type=model_type)
    elif model.lower() == "git":
        answer = get_git_vqa(raw_image, question, model_type=model_type)
    else:
        answer = generateAnswer(raw_image, question)
    return answer
