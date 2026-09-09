import os
from langchain_community.document_loaders import (PyPDFLoader,TextLoader,CSVLoader,Docx2txtLoader)
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk(filepath):

    extension = os.path.splitext(filepath)[1].lower()

    if extension == ".pdf":
        loader = PyPDFLoader(filepath)

    elif extension == ".txt":
        loader = TextLoader(filepath, encoding="utf-8")

    elif extension == ".csv":
        loader = CSVLoader(filepath)

    elif extension == ".docx":
        loader = Docx2txtLoader(filepath)

    else:
        raise ValueError(f"Unsupported file type: {extension}")
    
    pages = loader.load()

    splited_text = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", ". ", " ", ""])#splitting the text into chunks of 500 characters with an overlap of 50 characters.

    chunks = splited_text.split_documents(pages)


    print(f"Success! Broke the PDF into {len(chunks)} chunks.")
    return chunks
