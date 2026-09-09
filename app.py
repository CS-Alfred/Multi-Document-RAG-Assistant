import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from docload import load_and_chunk



load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_KEY")
groq_token = os.getenv("GROQ_API_KEY")




st.title("Multi-Document RAG Assistant")

st.write(
    "Upload a document and ask questions about its content."
)

uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf", "txt", "csv", "docx"]
)



@st.cache_resource
def get_embeddings():

    return HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
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




def create_vectorstore(uploaded_file):
    
    embeddings = get_embeddings()

    # Get the original file extension
    file_extension = os.path.splitext(
        uploaded_file.name
    )[1].lower()

    # Create a temporary file
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=file_extension
    ) as temp_file:

        temp_file.write(
            uploaded_file.getvalue()
        )

        temp_file_path = temp_file.name

    try:

        # Load and chunk the document
        chunks = load_and_chunk(
            temp_file_path
        )

        # Create Chroma vector store
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        return vectorstore, len(chunks)

    finally:

        # Always delete temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)




if uploaded_file:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    with st.spinner(
        "Reading and processing your document..."
    ):

        vectorstore, chunk_count = create_vectorstore(
            uploaded_file
        )

    st.success(
        f"Document processed successfully! "
        f"Created {chunk_count} chunks."
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
7. Use the retrieved context as the source of truth.

Context:

{context}

Question:

{question}

Answer:
"""


    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=[
            "context",
            "question"
        ]
    )




    llm = connect_llm()




    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,

        chain_type="stuff",

        retriever=vectorstore.as_retriever(
            search_kwargs={
                "k": 5
            }
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

        with st.spinner(
            "Searching the document..."
        ):

            result = qa_chain.invoke(
                {
                    "query": user_query
                }
            )




        st.subheader("Answer")

        st.write(
            result["result"]
        )




        st.subheader("Sources")

        for i, doc in enumerate(
            result["source_documents"]
        ):

            page_number = doc.metadata.get(
                "page",
                "N/A"
            )

            source_name = doc.metadata.get(
                "source",
                uploaded_file.name
            )

            st.write(
                f"**Source {i + 1}**"
            )

            st.write(
                f"File: `{os.path.basename(source_name)}`"
            )

            st.write(
                f"Page: `{page_number}`"
            )

            with st.expander(
                "View source content"
            ):

                st.write(
                    doc.page_content
                )