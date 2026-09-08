import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from pdfload import load_and_chunk

load_dotenv()#loading the api key from the .env file.

def creating_vector_db(chunks, persist_dir="./chroma_db"):

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")#model used for generating embeddings.

    vectorstorage = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=persist_dir
    )

    print(f"Success! Vector database created in the '{persist_dir}' folder.")
    return vectorstorage    

if __name__ == "__main__":
    my_chunks = load_and_chunk("sample.pdf")
    creating_vector_db(my_chunks)
