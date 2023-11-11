import cv2
import numpy
from transformers import BlipProcessor, BlipForConditionalGeneration


def get_caption(model, processor, image, text="a photography of", max_new_tokens=50, device="cpu"):
    inputs = processor(image, text, return_tensors="pt").to(device)
    caption_ids = model.generate(max_new_tokens=max_new_tokens, **inputs)
    return processor.decode(caption_ids[0], skip_special_tokens=True)


def run_BLIP(image):

    model_type = "large"

    processor = BlipProcessor.from_pretrained(
        f"Salesforce/blip-image-captioning-{model_type}"
    )
    model_cpu = BlipForConditionalGeneration.from_pretrained(
        f"Salesforce/blip-image-captioning-{model_type}"
    )

    # model_gpu = BlipForConditionalGeneration.from_pretrained(
    #     f"Salesforce/blip-image-captioning-{model_type}"
    # ).to(device)

    # img_path = os.path.join("../images/", "imhere1k.jpg")
    # raw_image = cv2.imread(img_path)
    raw_image = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
    cap = get_caption(model_cpu, processor, raw_image)
    return cap
