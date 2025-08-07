from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

import os
from dotenv import load_dotenv
load_dotenv()

# ✅ 1. Extract text (with OCR fallback)
def get_pdf_text(pdf_docs):
    full_text = ""

    for pdf_file in pdf_docs:
        pdf_reader = PdfReader(pdf_file)
        text = ""

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        # Fallback to OCR
        if not text.strip():
            pdf_file.seek(0)
            images = convert_from_bytes(pdf_file.read())
            for image in images:
                ocr_text = pytesseract.image_to_string(image)
                text += ocr_text

        full_text += text

    if not full_text.strip():
        raise ValueError("No extractable or OCR-readable text found in the PDF.")

    return full_text


# ✅ 2. Split text into chunks
def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks


# ✅ 3. Convert chunks into vector embeddings using FAISS
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vectorstore


# ✅ 4. Set up the LLM + retriever chain
def get_conversational_chain(vectorstore):
    llm = ChatOpenAI()
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )
    return chain
