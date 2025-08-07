import streamlit as st
from dotenv import load_dotenv
import os
from src.helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain
from langchain_community.chat_models import ChatGooglePalm
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

def main():
    st.set_page_config("Information Retrieval System 📚")
    st.header("Information-Retrival-System")

    user_question = st.text_input("Ask a question from the PDF:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    with st.sidebar:
        st.title("Upload PDF Files 📄")
        pdf_docs = st.file_uploader("Upload your PDFs here and click 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    st.write("✅ Extracted Text Length:", len(raw_text))

                    text_chunks = get_text_chunks(raw_text)
                    st.write("✅ Number of Chunks:", len(text_chunks))

                    vector_store = get_vector_store(text_chunks)
                    st.success("✅ Vector Store Created")

                    st.session_state.conversation = get_conversational_chain(vector_store)
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")

    if user_question and st.session_state.conversation:
        try:
            response = st.session_state.conversation({'question': user_question, 'chat_history': st.session_state.chat_history})
            st.session_state.chat_history = response['chat_history']

            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write("👤 You:", message.content)
                else:
                    st.write("🤖 Bot:", message.content)
        except Exception as e:
            st.error(f"❌ Error during Q&A: {str(e)}")

if __name__ == "__main__":
    main()
