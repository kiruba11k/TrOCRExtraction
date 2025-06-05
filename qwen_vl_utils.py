# from PIL import Image
# import requests
# from io import BytesIO

# def process_vision_info(messages):
#     """
#     Process visual input (image or video) from messages used in Qwen-VL model prompts.

#     Args:
#         messages (list): A list of messages in the Qwen-VL prompt format.

#     Returns:
#         tuple: (image_inputs, video_inputs)
#     """
#     image_inputs = []
#     video_inputs = []

#     for message in messages:
#         if "content" not in message:
#             continue
#         for item in message["content"]:
#             if item["type"] == "image":
#                 image = item["image"]
#                 # If image is a URL, fetch it
#                 if isinstance(image, str):
#                     response = requests.get(image)
#                     image = Image.open(BytesIO(response.content)).convert("RGB")
#                 image_inputs.append(image)
#             elif item["type"] == "video":
#                 # Placeholder: Extend here for video input handling if needed
#                 video_inputs.append(item["video"])

#     return image_inputs, video_inputs

from PIL import Image

def process_vision_info(messages):
    image_inputs, video_inputs = [], []
    for msg in messages:
        for content in msg["content"]:
            if content["type"] == "image":
                image = content["image"]
                if isinstance(image, str):  # if it's a URL
                    from PIL import Image
                    import requests
                    from io import BytesIO
                    response = requests.get(image)
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                elif isinstance(image, Image.Image):
                    image = image.convert("RGB")
                image_inputs.append(image)
            elif content["type"] == "video":
                video_inputs.append(content["video"])
    return image_inputs, video_inputs

