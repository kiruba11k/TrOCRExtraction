# import streamlit as st
# import torch
# import pandas as pd
# from PIL import Image
# from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info

# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "prithivMLmods/Qwen2-VL-OCR-2B-Instruct",
#     torch_dtype="auto",
#     device_map="auto"
# )
# processor = AutoProcessor.from_pretrained("prithivMLmods/Qwen2-VL-OCR-2B-Instruct")

# st.set_page_config(page_title="Image OCR - Qwen2-VL", layout="centered")
# st.title("Extract Name, Designation & Company from Image using Qwen2-VL-OCR")

# uploaded_files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# def run_qwen_ocr(image):
#     messages = [{
#         "role": "user",
#         "content": [
#             {"type": "image", "image": image},
#             {"type": "text", "text": "Extract name, designation, and company from this image."}
#         ]
#     }]
#     text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     image_inputs, video_inputs = process_vision_info(messages)
#     inputs = processor(
#         text=[text],
#         images=image_inputs,
#         videos=video_inputs,
#         padding=True,
#         return_tensors="pt"
#     ).to("cuda")

#     with torch.no_grad():
#         generated_ids = model.generate(**inputs, max_new_tokens=128)
#         trimmed_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
#         decoded = processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0]
#         return decoded

# def extract_fields(text):
#     lines = [line.strip() for line in text.split("\n") if line.strip()]
#     name, designation, company = "None", "None", "None"
#     if lines:
#         name = lines[0]
#         if len(lines) > 1:
#             designation = lines[1]
#         if len(lines) > 2:
#             company = lines[2]
#     return name, designation, company

# if uploaded_files:
#     results = []

#     for uploaded_file in uploaded_files:
#         image = Image.open(uploaded_file).convert("RGB")
#         st.image(image, caption=uploaded_file.name, use_column_width=True)

#         output_text = run_qwen_ocr(image)
#         name, designation, company = extract_fields(output_text)

#         results.append({
#             "Image": uploaded_file.name,
#             "Name": name,
#             "Designation": designation,
#             "Company": company
#         })

#     df = pd.DataFrame(results)
#     st.write("### Extracted Information")
#     st.dataframe(df)

#     tsv = df.to_csv(sep="\t", index=False).encode("utf-8")
#     st.download_button("Download as TSV", tsv, file_name="ocr_output.tsv")

import streamlit as st
import torch
from PIL import Image
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Title & Config
st.set_page_config(page_title="Qwen2-VL OCR", layout="centered")
st.title("🔍 Qwen2-VL OCR - Extract Name, Designation, Company")

# Load model and processor
try:
    st.info("⏳ Loading model and processor...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "prithivMLmods/Qwen2-VL-OCR-2B-Instruct",
        torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("prithivMLmods/Qwen2-VL-OCR-2B-Instruct")
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# File uploader
uploaded_files = st.file_uploader("📤 Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Extract fields from raw text
def extract_fields(text):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    name, designation, company = "None", "None", "None"
    if lines:
        name = lines[0]
        if len(lines) > 1:
            designation = lines[1]
        if len(lines) > 2:
            company = lines[2]
    return name, designation, company

# Processing uploaded images
if uploaded_files:
    results = []

    for file in uploaded_files:
        image = Image.open(file).convert("RGB")
        st.image(image, caption=file.name, use_column_width=True)

        # Prepare messages for Qwen2-VL
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Extract name, designation, and company from this image."}
            ]
        }]

        # Process input
        try:
            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True
            ).to(model.device)

            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            decoded_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            name, designation, company = extract_fields(decoded_text)

            results.append({
                "Image": file.name,
                "Name": name,
                "Designation": designation,
                "Company": company
            })

        except Exception as e:
            st.error(f"❌ Error processing {file.name}: {e}")

    # Display output
    df = pd.DataFrame(results)
    st.write("### 🧾 Extracted Data")
    st.dataframe(df)

    # TSV download
    tsv = df.to_csv(sep="\t", index=False).encode("utf-8")
    st.download_button("📥 Download as TSV", tsv, file_name="extracted_data.tsv")

