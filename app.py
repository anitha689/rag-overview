import glob
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

st.title("📄 My PDF Chatbot")

# Build the database once and remember it
@st.cache_resource
def build_db():
    docs = []
    for pdf in glob.glob("*.pdf"):
        docs.extend(PyPDFLoader(pdf).load())
    if not docs:
        return None, 0
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma.from_documents(chunks, embeddings), len(chunks)

db, num_chunks = build_db()
if db is None:
    st.error("No PDFs found in this folder.")
    st.stop()
st.caption(f"Searching {num_chunks} chunks from your PDFs")

llm = ChatOllama(model="llama3.2")

# Remember the chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat box
question = st.chat_input("Ask about your PDFs...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            results = db.similarity_search(question, k=3)
            context = "\n\n".join(r.page_content for r in results)
            prompt = f"Answer using only this context.\n\nContext:\n{context}\n\nQuestion: {question}"
            answer = llm.invoke(prompt).content
            sources = sorted({f"{r.metadata.get('source', 'unknown')}, page {r.metadata.get('page', 0) + 1}" for r in results})
            reply = answer + "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources)
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})