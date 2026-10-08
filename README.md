# AI-OCR-Document-Enhancement
AI-powered OCR system that extracts clear English text from images using image enhancement and Tesseract OCR.

import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import re

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

st.set_page_config(
    page_title="AI OCR",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>
.title {
    text-align:center;
    font-size:36px;
    font-weight:bold;
}
.sub {
    text-align:center;
    color:gray;
    margin-bottom:25px;
}
.box {
    padding:20px;
    border-radius:12px;
    background:#f5f7fa;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="title"> AI English OCR</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub">Extract clear English text from images</div>',
    unsafe_allow_html=True
)

file = st.file_uploader(
    "Upload your document/image",
    type=["jpg", "jpeg", "png"]
)

if file:
 
    pil_img = Image.open(file).convert("RGB")
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Resize
    img = cv2.resize(
        img, None, fx=2, fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    # Grayscale + contrast
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # Display
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        st.image(pil_img, use_container_width=True)

    with col2:
        st.subheader("Enhanced")
        st.image(enhanced, use_container_width=True)

    st.divider()

    if st.button("Extract English Text", use_container_width=True):

        with st.spinner("Extracting text..."):

            text = pytesseract.image_to_string(
                enhanced,
                lang="eng",
                config="--oem 3 --psm 3"
            )

            # Clean text
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n\s*\n+", "\n\n", text)
            text = text.strip()

        st.subheader("Extracted English Text")

        if text:
            st.text_area(
                "Result",
                text,
                height=250
            )

            st.download_button(
                "Download Text",
                text,
                "extracted_text.txt",
                "text/plain",
                use_container_width=True
            )
        else:
            st.warning("No text detected. Try a clearer image.")

else:
    st.info("Upload a clear English document to begin.")

st.divider()

st.caption(
    "AI-Based OCR with Automatic Document Image Enhancement"
)
