import os
from pypdf import PdfReader
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st

os.environ['OPENAI_API_KEY'] = '' # Put your OPENAI_API_KEY here
embeddings = OpenAIEmbeddings()

def chunk():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 10000,
        chunk_overlap = 500,
        separators = ["\n\n", "\n", " ", ""]
        
    )
    return text_splitter

llm = OpenAI(temperature=0.9, max_tokens=500) 

st.title("DocuAnswer 📄✅")
st.sidebar.title("Documents")
doc = st.sidebar.file_uploader("Upload your document")

process_doc = st.sidebar.button("Teach me the document!")

if process_doc and doc is not None:
    # load doc
    uploaded_file_path = "uploaded_document.pdf"
    with open(uploaded_file_path, "wb") as f:
        f.write(doc.read())

    # Load the document
    loader = PyPDFLoader(uploaded_file_path)
    data = loader.load()

    # chunking
    pages = chunk().split_documents(data)

    # embeddings
    openai_vectorindex = FAISS.from_documents(pages, embeddings)
    openai_vectorindex.save_local("faiss_store")

    

question = st.text_input("Question")
process = st.button("Ask me!")
try:
    if process or question:
        
        vector_index = FAISS.load_local("faiss_store", embeddings)
        # chain
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_index.as_retriever())
        result = chain({"question": question}, return_only_outputs=True)

        st.subheader(result["answer"])
except:
    st.subheader("An error has occured. Either your API key has expired in uses or you asked too many questions too fast. Try again after 1 minute!")