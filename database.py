import os
import shutil
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from pdfload import load_and_chunk

load_dotenv()#loading the api key from the .env file.

def creating_vector_db(chunks, persist_dir="./chroma_db"):
    hf_token = os.getenv("HUGGINGFACEHUB_API_KEY")
    embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2",huggingfacehub_api_token=hf_token)#model used for generating embeddings.
    
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    vectorstorage = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=persist_dir
    )

    print(f"Success! Vector database created in the '{persist_dir}' folder.")
    return vectorstorage    

