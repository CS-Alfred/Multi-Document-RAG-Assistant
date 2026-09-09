import os
import streamlit as st
import tempfile
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from pdfload import load_and_chunk


load_dotenv()

hf_token =os.getenv("HUGGINGFACEHUB_API_KEY")#loading the api key from the .env file.
groq_token = os.getenv("GROQ_API_KEY")#loading the api key from the .env file.


st.title("Document Reader")
st.write("Ask questions and get answers directly from your PDF!")

uploaded_file = st.file_uploader("Upload your document",type=["pdf", "txt", "csv", "docx"])#user uploading the pdf file.



@st.cache_resource
def get_embeddings():

    return HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",#creating the embeddings using the sentence-transformers/all-MiniLM-L6-v2 model.
        huggingfacehub_api_token=hf_token
    )



@st.cache_resource
def connect_llm():

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=groq_token,
        temperature=0.2,
        max_tokens=512
    )

    return llm




def create_vectorstore(uploaded_file):#creating the vector database from the uploaded pdf file.

    embeddings = get_embeddings()

    # Create temporary PDF file
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.getvalue())

        temp_pdf_path = temp_file.name


    # Load and chunk PDF
    chunks = load_and_chunk(temp_pdf_path)


    # Create Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )


    # Delete temporary PDF
    os.remove(temp_pdf_path)


    return vectorstore, len(chunks)




if uploaded_file:

    st.success(f"Uploaded: {uploaded_file.name}")

    with st.spinner("Reading and processing your PDF..."):

        vectorstore, chunk_count = create_vectorstore(
            uploaded_file
        )

    st.success(
        f"PDF processed successfully!"
    )




    prompt_template = """
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the information
contained in the provided context.

Rules:

1. Understand the question before answering.
2. Do not simply repeat matching passages.
3. Synthesize information from multiple passages when necessary.
4. Give a clear and direct answer.
5. If the document does not contain enough information,
   say that the answer cannot be found in the document.
6. Do not invent information.
7. When possible, mention the relevant page number.

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


    llm = connect_llm()


    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",

        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 5}
        ),

        return_source_documents=True,

        chain_type_kwargs={
            "prompt": PROMPT
        }
    )




    user_query = st.text_input(
        "What would you like to know about the document?"
    )


    if user_query:

        with st.spinner("Searching the document..."):#query chain is invoked to get the answer from the document.

            result = qa_chain.invoke({
                "query": user_query
            })




        st.subheader("Answer")

        st.write(result["result"])




        st.subheader("Sources")

        for i, doc in enumerate(
            result["source_documents"]
        ):

            page_number = (
                doc.metadata.get("page", "Unknown")
            )

            st.write(
                f"**Source {i + 1} — Page {page_number}**"
            )

            st.write(doc.page_content)