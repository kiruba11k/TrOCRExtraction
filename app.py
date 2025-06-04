import streamlit as st
import torch
import os
import io
import pandas as pd
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torchvision.transforms as transforms

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

# 
def preprocess_image(image):
    image = image.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

def extract_text(image_tensor):
    with torch.no_grad():
        generated_ids = model.generate(image_tensor)
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

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

# Streamlit UI
st.set_page_config(page_title="Image to Excel - TrOCR", layout="centered")
st.title("OCR with TrOCR - Extract Name, Designation, Company")

uploaded_files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    results = []

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=uploaded_file.name, use_column_width=True)

        tensor = preprocess_image(image)
        text = extract_text(tensor)
        name, designation, company = extract_fields(text)
        results.append({
            "Image": uploaded_file.name,
            "Name": name,
            "Designation": designation,
            "Company": company
        })

    df = pd.DataFrame(results)
    st.write("### Extracted Data")
    st.dataframe(df)

    # TSV Download
    tsv = df.to_csv(sep="\t", index=False).encode("utf-8")
    st.download_button(" Download as Excel TSV", tsv, file_name="output.tsv")

