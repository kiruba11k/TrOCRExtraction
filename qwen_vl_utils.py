
from PIL import Image

def process_vision_info(messages):
    image_inputs = []
    video_inputs = []

    for message in messages:
        for content in message["content"]:
            if content["type"] == "image":
                image = content["image"]
                if isinstance(image, Image.Image):
                    image_inputs.append(image)
                elif isinstance(image, str):
                    raise ValueError("Expected PIL.Image.Image, got string.")
            elif content["type"] == "video":
                video_inputs.append(content["video"])

    return image_inputs, video_inputs
