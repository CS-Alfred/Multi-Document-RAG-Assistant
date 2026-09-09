import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


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
    # We use llama-3.3-70b-versatile
    M = ChatGroq(
        model="openai/gpt-oss-120b", 
        api_key=groq_token,
        temperature=0.2,
        max_tokens=512
    )
    return M
    

llm = connect_llm()

@st.cache_resource
def get_qa_chain(_llm, _vectorstore):

    prompt_template = """
You are a helpful Bible document assistant.

Answer the user's question using ONLY the information contained
in the provided context.

Important instructions:
1. Understand the user's question before answering.
2. Do not simply list passages that contain the searched word.
3. Synthesize information from multiple passages when necessary.
4. If a person has the same name as another person in the Bible,
   identify the correct person from the context.
5. For questions such as "Who is Joseph?", give a concise
   identification of the person and explain his important role
   or relationships.
6. If the context does not contain enough information to answer,
   say that the information is not sufficient.
7. Do not make up information that is not supported by the context.

Context:
{context}

Question:
{question}

Answer:
"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=_llm,
        chain_type="stuff",
        retriever=_vectorstore.as_retriever(
            search_kwargs={"k": 5}
        ),
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PROMPT
        }
    )

qa_chain = get_qa_chain(llm, vectorstore)


user_query = st.text_input("What would you like to know about the document?")#user asking the query

if user_query:
    with st.spinner("Searching..."):
        result = qa_chain.invoke({"query": user_query})

        st.success("Your answer is ready!")

        st.write("### Answer")
        st.write(result["result"])

        st.write("### Sources")

        for i, doc in enumerate(result["source_documents"]):
            st.write(f"**Source {i + 1}**")
            st.write(doc.page_content)
