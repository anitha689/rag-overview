# 📄 PDF Chatbot (RAG)

A chatbot that answers questions about your own PDF documents, built with Retrieval-Augmented Generation (RAG). It runs 100% locally and free using Ollama.

## Features
- Chat with multiple PDFs at once
- Hybrid search: semantic (meaning) + keyword (BM25) combined with Reciprocal Rank Fusion
- Shows the source PDF and page for every answer
- Chat interface built with Streamlit

## How it works
1. **Load** all PDFs in the folder
2. **Chunk** the text into ~800-character pieces
3. **Embed** the chunks with `nomic-embed-text` and store them in **Chroma**
4. **Retrieve** the best chunks using hybrid search
5. **Generate** an answer with **Llama 3.2**, using only the retrieved text

## Tech stack
Python · LangChain · Chroma · Ollama (Llama 3.2, nomic-embed-text) · BM25 · Streamlit

## Setup
1. Install [Ollama](https://ollama.com), then pull the models:
```
   ollama pull llama3.2
   ollama pull nomic-embed-text
```
2. Install the Python packages:
```
   pip install -r requirements.txt
```
3. Put your PDFs in the project folder.

## Run
```
python -m streamlit run app.py
```
