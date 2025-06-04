import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from io import BytesIO
from typing import TypedDict

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph


processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Add GROQ_API_KEY to `.streamlit/secrets.toml`")
    st.stop()

class MyState(TypedDict):
    image: BytesIO
    text: str
    output: str


llm = ChatGroq(
    model="llama3-8b-8192",  
    groq_api_key=GROQ_API_KEY,
    temperature=0.2,
)

def ocr_step(state: MyState) -> MyState:
    uploaded_file = state.get("image")
    if uploaded_file is None:
        raise ValueError("Missing 'image' in state.")

    image = Image.open(uploaded_file).convert("RGB")

    image = image.resize((1024, 1024))

    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    state["text"] = text.strip()
    return state

def ner_step(state: MyState) -> MyState:
    text = state["text"]
    prompt = f"""
Extract Name, Designation, and Company from this text and return only tab-separated values. One entry per line. No explanation.

Text:
{text.strip()}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    state["output"] = response.content.strip()
    return state

workflow = StateGraph(state_schema=MyState)
workflow.add_node("OCR", ocr_step)
workflow.add_node("NER", ner_step)
workflow.set_entry_point("OCR")
workflow.add_edge("OCR", "NER")
workflow.set_finish_point("NER")
graph = workflow.compile()

st.title("OCR Extraction using TrOCR")

uploaded_files = st.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    results = []
    for uploaded_file in uploaded_files:
        st.image(uploaded_file, width=250)
        with st.spinner("Extracting..."):
            file_copy = BytesIO(uploaded_file.read())
            file_copy.seek(0)
            result = graph.invoke({"image": file_copy})
            results.append(result["output"])

    st.markdown("### Extracted TSV Output")
    result_text = "\n".join(results)
    st.code(result_text, language="tsv")
    st.download_button("Download TSV", result_text, file_name="entities.tsv")
else:
    st.info("Upload at least one image to begin.")
