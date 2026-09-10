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