# Multi-Document RAG Assistant
### This project is an AI powered document question answering application built with Python, LangChain, ChromaDB, Hugging Face, Groq, and Streamlit using Retrieval-Augmented Generation (RAG).


## Overview

This application uses a Retrieval-Augmented Generation (RAG) pipeline to retrieve relevant information from uploaded documents and provide grounded answers using a Large Language Model (LLM).

Instead of relying only on the LLM's pretrained knowledge, the system retrieves relevant document content and provides it as context to the model before generating an answer. This helps make responses more relevant to the uploaded documents and reduces unsupported answers.

## Video Demo

[Watch the Multi-Document RAG Assistant Demo](https://youtu.be/iIwzZryBjT0?si=NnEqiRDAezm9pOLO)

## Features
* Ask natural language questions about uploaded documents.
* Retrieves relevant document content before generating an answer.
* Uses Hugging Face sentence transformer embeddings to represent document content as vectors.
* Uses ChromaDB for storing and retrieving document embeddings.
* The LLM is instructed to answer using the retrieved document context rather than relying solely on general knowledge.
* Retrieved document sections can be displayed alongside generated answers.
* Provides a simple web interface for uploading documents and asking questions.

## Technologies Used
* Python
* LangChain
* ChromaDB
* Hugging Face
* Groq
* Streamlit

## Basic Requirements
* Python 3.10+
* Groq API key
* Hugging Face API key

**Python dependencies are listed in** ***Requirements.txt***

## Installation
1. Clone the repository
    - git clone https://github.com/CS-Alfred/Multi-Document-RAG-Assistant.git
        1. Navigate into project folder
            - cd Multi-Document-RAG-Assistant
2. Create a virtual environment
    - python -m venv venv
        1. To activate in windows
            - venv\Scripts\activate
        2. To activate in macOS/Linux
            - source venv/bin/activate
3. Install Dependencies
    - pip install -r Requirements.txt

4. Configure API Keys
    - Create a .env file in the project root
    **HUGGINGFACEHUB_API_KEY=your_huggingface_api_key**
    **GROQ_API_KEY=your_groq_api_key**

    ***Replace the placeholder values with your own API keys.***   

    ***Important: Never commit your .env file or expose your API keys publicly.***     

5. Run the Application
    - streamlit run app.py

    **The application will open in your browser.**

## References and Resources
This project was developed by studying and referring to documentation, tutorials, videos, and resources related to RAG, LangChain, vector databases, embeddings, and LLM applications.

### Documentation
* LangChain Documentation
* ChromaDB Documentation
* Hugging Face Documentation
* Groq Documentation
* Streamlit Documentation

### YouTube
* [Large Language Models explained briefly](https://www.youtube.com/watch?v=LPZh9BOjkQs&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=5)
* [Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=6)
* [Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=7)

## Author
[C S Alfred](https://github.com/CS-Alfred)