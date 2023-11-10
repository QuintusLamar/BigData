import os
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration

# conditional image captioning
def get_caption(
    model, processor, image, text="a photography of", max_new_tokens=50, device="cpu"
):
    inputs = processor(image, text, return_tensors="pt").to(device)
    caption_ids = model.generate(max_new_tokens=max_new_tokens, **inputs)
    return processor.decode(caption_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    model_type = "large"
    device = "cpu"

    processor = BlipProcessor.from_pretrained(
        f"Salesforce/blip-image-captioning-{model_type}"
    )
    model_cpu = BlipForConditionalGeneration.from_pretrained(
        f"Salesforce/blip-image-captioning-{model_type}"
    )

    model_gpu = BlipForConditionalGeneration.from_pretrained(
        f"Salesforce/blip-image-captioning-{model_type}"
    ).to(device)

    img_path = os.path.join("Images", "imhere1k.jpg")
    raw_image = cv2.imread(img_path)
    cap = get_caption(model_cpu, processor, raw_image)
    print(cap)
