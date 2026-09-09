import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings


load_dotenv()

hf_token =os.getenv("HUGGINGFACEHUB_API_KEY")#loading the api key from the .env file.


st.title("Document Reader")
st.write("Ask questions and get answers directly from your PDF!")

@st.cache_resource

def conect_vector_db():
    embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2",huggingfacehub_api_token=hf_token)
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)#Establishing the connection between the vector database and the webapp.

    return db


vectorstore = conect_vector_db()


user_query = st.text_input("What would you like to know about the document?")#user asking the query

if user_query:
    with st.spinner("Searching..."):
        docs = vectorstore.similarity_search(user_query, k=3)

        st.success()

        for i, doc in enumerate(docs):
            st.write(doc.page_content)

