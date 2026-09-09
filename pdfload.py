from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk(filepath):

    loader = PyPDFLoader(filepath) #loading the pdf file.
    pages = loader.load()

    splited_text = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", ". ", " ", ""])#splitting the text into chunks of 500 characters with an overlap of 50 characters.

    chunks = splited_text.split_documents(pages)


    print(f"Success! Broke the PDF into {len(chunks)} chunks.")
    return chunks
