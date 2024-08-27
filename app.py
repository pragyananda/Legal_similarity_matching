import re
import numpy as np
import streamlit as st
import pytesseract
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer


def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def extract_text_from_scanned_pdf(pdf_file):
    text = ""
    images = convert_from_path(pdf_file)
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower().strip()


model = SentenceTransformer("all-MiniLM-L6-v2")

st.title('PDF Similarity Checker')

uploaded_file1 = st.file_uploader("Choose the first PDF file", type="pdf")
uploaded_file2 = st.file_uploader("Choose the second PDF file", type="pdf")

if uploaded_file1 and uploaded_file2:
    text_1 = extract_text_from_pdf(uploaded_file1)
    if not text_1.strip():
        text_1 = extract_text_from_scanned_pdf(uploaded_file1)
    cleaned_text_1 = clean_text(text_1)
    
    text_2 = extract_text_from_pdf(uploaded_file2)
    if not text_2.strip():
        text_2 = extract_text_from_scanned_pdf(uploaded_file2)
    cleaned_text_2 = clean_text(text_2)
    
    sentence_embeddings_1 = model.encode([cleaned_text_1])
    sentence_embeddings_2 = model.encode([cleaned_text_2])
    
    similarity = np.inner(sentence_embeddings_1, sentence_embeddings_2)[0][0]
    similarity_percentage = similarity * 100
    
    st.write("Similarity Percentage: {:.2f}%".format(similarity_percentage))
