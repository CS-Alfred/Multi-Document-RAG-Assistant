import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA


load_dotenv()

hf_token =os.getenv("HUGGINGFACEHUB_API_KEY")#loading the api key from the .env file.
groq_token = os.getenv("GROQ_API_KEY")#loading the api key from the .env file.


st.title("Document Reader")
st.write("Ask questions and get answers directly from your PDF!")

@st.cache_resource

def conect_vector_db():
    embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2",huggingfacehub_api_token=hf_token)
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)#Establishing the connection between the vector database and the webapp.

    return db


vectorstore = conect_vector_db()

@st.cache_resource

def connect_llm():
    # We use Llama 3.1
    return ChatGroq(
        repo_id="llama-3.1-8b-instant", 
        huggingfacehub_api_token=hf_token,
        temperature=0.1,
        max_new_tokens=512
    )
    

llm = connect_llm()

@st.cache_resource

def get_qa_chain(_llm, _vectorstore):
    return RetrievalQA.from_chain_type(
        llm=_llm,
        chain_type="stuff",
        retriever=_vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

qa_chain = get_qa_chain(llm, vectorstore)


user_query = st.text_input("What would you like to know about the document?")#user asking the query

if user_query:
    with st.spinner("Searching..."):
        result = qa_chain.invoke({"query": user_query})

        st.success("Your answer is ready!")

        for i, doc in enumerate(result["source_documents"]):
            st.write(doc.page_content)

