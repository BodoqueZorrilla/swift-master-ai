from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os

# Configuration
PDF_DIR = "./pdfs"           # Put your PDFs here
DB_DIR = "./chroma_db"       # Vector database persists here

def ingest_documents():
    # Load all PDFs from directory
    loader = DirectoryLoader(
        PDF_DIR, 
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from PDFs")

    swift_loader = DirectoryLoader(
        "../../ElectroMX/electromx-ios-app-admin/ElectroMXAdmin", 
        glob="**/*.swift",
        loader_cls=TextLoader
    )
    swift_docs = swift_loader.load()
    documents.extend(swift_docs)

    # Split into chunks (important for code/design docs)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # Create embeddings using local Ollama model
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )

    # Store in ChromaDB (persistent, offline)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name="swift_knowledge"
    )
    vectorstore.persist()
    print(f"✅ Ingested {len(chunks)} chunks into {DB_DIR}")

if __name__ == "__main__":
    os.makedirs(PDF_DIR, exist_ok=True)
    ingest_documents()