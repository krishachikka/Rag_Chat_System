import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
import re

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")  # fetch API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # configure Google AI

# Extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

# Split large text into smaller chunks for embeddings
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Create and save FAISS vector store
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Prepare QA chain with Google Generative AI
def get_conversational_chain():
    prompt_template = """
    Answer the question from the context below in **short, simple, and easy-to-understand points**. 
    Use bullet points. 
    If the answer is not in the context, say "Answer is not available in the context". 
    Do not provide wrong information.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Generate response for a user query
def generate_response(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    # Remove any unwanted '*' in the output
    raw_text = response["output_text"]
    cleaned_response = re.sub(r'(?<!\S)\*+(?!\S)', '\n•', raw_text).strip()
    return cleaned_response

def main():
    st.set_page_config("Chat with your pdf!")

    # Inject custom CSS (kept exactly as your original)
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap');

        * {
            font-family: 'Poppins', sans-serif;
        }

        /* Background */
        .stApp {
            background-color: #1b2021;
            color: #ffd9da;
        }

        /* Header */
        h1, h2, h3 {
            color: #ffd9da;
            text-align: center;
            font-weight: 600;
        }

        /* Chat container */
        .chat-container {
            # background-color: #30343f;
            # border-radius: 15px;
            # padding: 20px;
            # overflow-y: auto;
            # box-shadow: 0 0 15px rgba(234,99,140,0.3);
            # margin-bottom: 20px;
            # z-index: -1;
        }

        /* User bubble */
        .user-bubble {
            background-color: #89023e;
            color: #fff;
            padding: 12px 16px;
            border-radius: 18px 18px 0 18px;
            display: inline-block;
            margin: 10px 0;
            float: right;
            max-width: 75%;
            word-wrap: break-word;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
        }

        /* Bot bubble */
        .bot-bubble {
            background-color: #ffd9da;
            color: #1b2021;
            padding: 12px 16px;
            border-radius: 18px 18px 18px 0;
            display: inline-block;
            margin: 10px 0;
            float: left;
            max-width: 75%;
            word-wrap: break-word;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }

        /* Clear floats */
        .clearfix::after {
            content: "";
            clear: both;
            display: table;
        }

        /* Input */
        input[type="text"] {
            border-radius: 25px !important;
            border: 2px solid #ea638c;
            padding: 12px;
            background-color: #30343f;
            color: #ffd9da;
            z-index: 10;
            position: sticky;
            bottom: 0;
        }

        /* Button */
        button[kind="secondaryFormSubmit"] {
            background-color: #ea638c;
            color: #fff;
            border-radius: 25px;
            font-weight: 600;
            border: none;
            padding: 8px 16px;
        }
        button[kind="secondaryFormSubmit"]:hover {
            background-color: #89023e;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #89023e;
            color: #ffd9da;
            border-right: 2px solid #89023e;
        }

        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([0.1, 0.9])

    with col1:
        st.image("icon.jpg", width=60)

    with col2:
        st.markdown("""
        <h2 style="color:#ffd9da; font-family: 'Poppins', sans-serif; font-weight:600; margin:0; text-align:left;">
            Chat with your pdf!
        </h2>
        """, unsafe_allow_html=True)



    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar UI for PDF upload
    with st.sidebar:
        st.markdown("""
        <style>
        [data-testid="stSidebar"] {
            background-color: #30343f;
            color: #ffd9da;
        }

        [data-testid="stFileUploader"] {
            border: 2px dashed #ea638c;
            background-color: #1b2021;
            border-radius: 12px;
            padding: 20px;
            transition: all 0.3s ease;
        }

        [data-testid="stFileUploader"]:hover {
            border-color: #ffd9da;
            box-shadow: 0 0 12px rgba(234, 99, 140, 0.5);
        }

        [data-testid="stFileUploader"] label div {
            color: #ffd9da !important;
            font-family: 'Poppins', sans-serif;
            font-weight: 500;
            text-align: center;
        }

        [data-testid="stFileUploader"] section div {
            color: #ea638c !important;
            font-family: 'Poppins', sans-serif;
        }

        div.stButton > button {
            background-color: black;
            color: #ffd9da;
            border-radius: 25px;
            border: none;
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            transition: 0.3s;
        }

        div.stSucess > div {
            background-color: #1b2021;
            color: #ffd9da !important;
        }

        div.stButton > button:hover {
            background-color: #ea638c;
            color: #1b2021;
            transform: scale(1.05);
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)

        st.title("Menu")
        pdf_docs = st.file_uploader("Upload PDF(s) and click 'Process'", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("PDF processed successfully!")

    # Display chat messages
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='clearfix'><div class='user-bubble'>{msg['content']}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='clearfix'><div class='bot-bubble'>{msg['content']}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Chat input
    with st.form(key="chat_form", clear_on_submit=True):
        user_question = st.text_input("Ask a question:", "")
        submit = st.form_submit_button("Send")

    # Handle user input
    if submit and user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.spinner("Thinking..."):
            reply = generate_response(user_question)
        st.session_state.messages.append({"role": "bot", "content": reply})
        st.rerun()

if __name__ == "__main__":
    main()
