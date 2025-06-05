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
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import io

# Load model & processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "prithivMLmods/Qwen2-VL-OCR-2B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("prithivMLmods/Qwen2-VL-OCR-2B-Instruct")

st.title("Name, Designation & Company Extractor (Qwen2-VL)")
st.caption("Upload image(s) like business cards, ID cards, or resumes")

uploaded_files = st.file_uploader("Upload Image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    results = []

    for uploaded_file in uploaded_files:
        # Load image
        image = Image.open(uploaded_file).convert("RGB")

        # Display image
        st.image(image, caption=uploaded_file.name, use_column_width=True)

        # Prepare input
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        buffered.seek(0)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Extract name, designation, and company from this image."}
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Basic formatting (optional: regex parsing here)
        st.markdown("### Extracted Info:")
        st.text(output_text)

        # Convert to TSV if possible
        name, designation, company = [""] * 3
        lines = output_text.split("\n")
        for line in lines:
            if "name" in line.lower():
                name = line.split(":")[-1].strip()
            elif "designation" in line.lower() or "role" in line.lower():
                designation = line.split(":")[-1].strip()
            elif "company" in line.lower() or "organization" in line.lower():
                company = line.split(":")[-1].strip()

        tsv_output = f"{name}\t{designation}\t{company}"
        st.markdown("### Tab-separated Output")
        st.code(tsv_output, language="tsv")

        results.append(tsv_output)

    # Final download
    if results:
        tsv_all = "\n".join(results)
        st.download_button(
            label=" Download TSV",
            data=tsv_all,
            file_name="extracted_data.tsv",
            mime="text/tsv",
        )
