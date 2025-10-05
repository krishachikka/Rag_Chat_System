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

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

# Function to split extracted text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save a vector store using FAISS
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to define the conversational chain
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

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Function to handle user input, search for similar documents, and generate a response
def generate_response(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load FAISS index
    new_db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )
    return response["output_text"]

# Main function
def main():
    st.set_page_config("Legal Document Chat")
    st.header("💬 Chat with your Legal Document")

    # Keep chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar file upload
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader(
            "Upload PDF(s) and click 'Process'", accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("PDF processed successfully!")

    # Chat history UI
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f"<div style='text-align:right; background-color:#DCF8C6; color:black; padding:8px; border-radius:10px; margin:5px'>{msg['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='text-align:left; background-color:white; color:black; padding:8px; border-radius:10px; margin:5px'>{msg['content']}</div>",
                unsafe_allow_html=True,
            )

    # Input box and send button
    with st.form(key="chat_form", clear_on_submit=True):
        user_question = st.text_input("Ask a question:", "")
        submit = st.form_submit_button("Send")

    if submit and user_question:
        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_question})

        with st.spinner("🤖 Thinking..."):
            reply = generate_response(user_question)

        # Save bot reply
        st.session_state.messages.append({"role": "bot", "content": reply})

        # Refresh page so new messages show
        st.rerun()

# Run app
if __name__ == "__main__":
    main()
