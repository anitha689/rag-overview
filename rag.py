import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

# 1. Load ALL PDFs in the folder
docs = []
for pdf in glob.glob("*.pdf"):
    print("Loading:", pdf)
    docs.extend(PyPDFLoader(pdf).load())
print(f"Loaded {len(docs)} pages from all PDFs")

if not docs:
    print("No PDFs found! Put some PDFs in this folder and try again.")
    exit()

# 2. Split it into small chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks")

# 3. Turn chunks into numbers and store them in Chroma
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma.from_documents(chunks, embeddings)

# 4. Ask questions
llm = ChatOllama(model="llama3.2")
while True:
    question = input("\nAsk a question (or type quit): ")
    if question.lower() == "quit":
        break
    results = db.similarity_search(question, k=3)
    context = "\n\n".join(r.page_content for r in results)
    prompt = f"Answer using only this context.\n\nContext:\n{context}\n\nQuestion: {question}"
    answer = llm.invoke(prompt)
    print("\nAnswer:", answer.content)

    # NEW: show where the answer came from
    print("\nSources:")
    shown = set()
    for r in results:
        source = r.metadata.get("source", "unknown")
        page = r.metadata.get("page", 0) + 1
        if (source, page) not in shown:
            shown.add((source, page))
            print(f"  - {source}, page {page}")