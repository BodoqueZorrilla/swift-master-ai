# 🧠 Swift AI Teacher

A local, privacy‑first AI assistant that answers Swift coding questions, reviews your code, and provides context‑aware feedback – all running on your own machine using Ollama, RAG (Retrieval‑Augmented Generation), and a lightweight vector database.

## ✨ Features

- **Local LLM** – Uses `gemma4:12b` (or any Ollama model) – no API keys, no data leaving your computer.
- **RAG with your documents** – Load PDFs (tutorials, Swift books, WWDC transcripts) to give the model extra context.
- **Swift file support** – Index `.swift` files from your own projects for project‑specific queries.
- **Code review assistance** – Ask things like *"Review my NetworkManager.swift for Swift 6 concurrency issues"*.

## 📋 Prerequisites

- **macOS / Linux / Windows** (Ollama works on all major platforms)
- **Python 3.10+**
- **Git** (to clone the repository)

## 🚀 Installation

### 1. Install and start Ollama

Follow the instructions at [ollama.com](https://ollama.com) to install Ollama on your system.

Then pull the required models:

```bash
ollama pull gemma4:12b
ollama pull nomic-embed-text   # local embedding model for RAG
```

> 💡 You can replace `gemma4:12b` with any other Ollama model (e.g., `llama3.2`, `qwen2.5`). Adjust the model name in `swift_master.py` accordingly.

### 2. Clone this repository

```bash
git clone https://github.com/your-username/swift-ai-teacher.git
cd swift-ai-teacher
```

### 3. Create a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install ollama langchain langchain-community chromadb pypdf sentence-transformers streamlit
```

- **ollama** – to interact with local models.
- **langchain** – orchestration (optional but convenient).
- **chromadb** – lightweight vector database.
- **sentence-transformers** – local embedding model (`all-MiniLM-L6-v2` works great).
- **pypdf** – to extract text from PDFs.
- **streamlit** – optional UI.

## 📚 Ingesting your knowledge base (RAG)

The AI can answer questions based on your own documents. Place any PDF files (e.g., Swift books, articles, notes) inside the `./docs/` folder (create it if it doesn't exist).

Then run the ingestion script:

```bash
python3 ingest.py
```

This will:

1. Read all PDFs from `./docs/`
2. Split them into chunks
3. Create embeddings (using the local `nomic-embed-text` model or sentence-transformers)
4. Store them in a Chroma vector database

> 🔄 Whenever you add, update, or remove a PDF, re‑run `ingest.py` to refresh the vector database.

## 🔧 Adding your own Swift code files

To let the AI review your actual Swift projects, extend `ingest.py` as follows:

```python
from langchain_community.document_loaders import TextLoader, DirectoryLoader

# Add this to ingest.py – it also loads .swift files
swift_loader = DirectoryLoader(
    "./my_project",          # 👈 change to your project folder
    glob="**/*.swift",
    loader_cls=TextLoader
)
swift_docs = swift_loader.load()
documents.extend(swift_docs)
```

After adding this, re‑run `ingest.py` and the AI will have context from your Swift files.

## 🧠 Running the AI Teacher

Make sure Ollama is running in the background:

```bash
ollama serve
```

Then start the main script:

```bash
python3 swift_master.py
```

This launches a terminal‑based chat where you can ask Swift questions. The system will automatically retrieve relevant chunks from your ingested documents and Swift files to provide accurate, context‑aware answers.

## 🖥️ Optional: Web UI with Streamlit

For a more user‑friendly experience, you can run the Streamlit interface:

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`.

## 🌐 Optional: Open WebUI (ChatGPT‑style interface)

For a full ChatGPT‑like experience with built‑in RAG support, you can run **Open WebUI** via Docker:

```bash
docker run -d -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

Then visit `http://localhost:3000`. You can upload your PDFs directly in the chat and use `gemma4:12b` as your model.

## 🧹 Maintenance

Remove unused Ollama models to free disk space:

```bash
ollama list
ollama rm <model_name>
ollama prune
```

## 🤝 Contributing

Feel free to open issues or submit pull requests to improve the teacher – better prompts, support for more file types, or performance improvements are always welcome.

## 📄 License

Bodoque – use it, modify it, teach Swift with it.
