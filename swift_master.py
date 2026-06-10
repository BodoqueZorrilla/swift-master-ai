#!/usr/bin/env python3
"""
Swift AI Teacher - with full project awareness.
Commands:
  - Ask any Swift question (RAG answers using your PDFs + codebase)
  - create: <request>      → generates new .swift file (e.g., create: a generic API client)
  - improve: <request>     → improves an existing .swift file (e.g., improve: NetworkManager.swift)
  - exit
"""

import os
import re
import subprocess
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# ========== CONFIGURATION ==========
MODEL_NAME = "gemma4:12b"               # Change to any Ollama model
EMBEDDING_MODEL = "nomic-embed-text"    # Must be pulled locally
CHROMA_DIR = "./chroma_db"
DOCS_DIR = "./pdfs"                     # PDFs go here
PROJECT_DIR = "../../ElectroMX/electromx-ios-app-admin/ElectroMXAdmin" # Path to your active project

# ========== ENGINE WRAPPER TO PREVENT PYDANTIC COLLISION ==========
class SwiftAIEngine:
    """
    Lightweight wrapper that delegates queries to the RetrievalQA chain
    while safely exposing the raw LLM for custom generation tasks.
    """
    def __init__(self, qa, llm):
        self.qa = qa
        self.llm = llm

    def invoke(self, *args, **kwargs):
        return self.qa.invoke(*args, **kwargs)

# ========== EMBEDDING & VECTOR STORE ==========
def get_vectorstore():
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
    if os.path.exists(CHROMA_DIR):
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
    else:
        print("⚠️ No vector database found. Please run `python3 ingest.py` first.")
        return None

# ========== LOAD QA CHAIN ==========
def load_qa():
    vectorstore = get_vectorstore()
    if not vectorstore:
        return None
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    llm = OllamaLLM(model=MODEL_NAME)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return SwiftAIEngine(qa, llm)

# ========== PROJECT FILE UTILITIES ==========
def list_swift_files(directory):
    """Return list of all .swift files recursively."""
    swift_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(".swift"):
                swift_files.append(os.path.join(root, f))
    return swift_files

def read_swift_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"// Error reading file: {e}"

def write_swift_file(filepath, content):
    """Create or overwrite a Swift file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Written to {filepath}")

# ========== SMART CONTEXT RETRIEVAL ==========
def retrieve_relevant_code(qa, request, project_dir):
    """
    Given a user request (e.g. "create a ViewModel based on @WebsocketModel.swift"),
    retrieve related Swift files from the project using RAG.
    Also extract the specific file if mentioned with @Filename.swift.
    """
    # 1. Check for explicit @filename references
    pattern = r'@(\w+\.swift)'
    mentioned_files = re.findall(pattern, request)
    explicit_content = ""
    for fname in mentioned_files:
        # Search recursively for the mentioned file
        filepath = None
        for root, _, files in os.walk(project_dir):
            if fname in files:
                filepath = os.path.join(root, fname)
                break
        
        if filepath and os.path.exists(filepath):
            content = read_swift_file(filepath)
            explicit_content += f"\n\n// ----- {fname} -----\n{content}\n"
        else:
            print(f"⚠️ Mentioned file {fname} not found recursively inside {project_dir}")

    # 2. Use RAG to find other relevant code snippets (classes, patterns)
    rag_context = ""
    if qa:
        # Query the vector store for Swift code relevant to the request
        rag_result = qa.invoke(f"Swift code related to: {request}")
        rag_context = rag_result["result"]
        # Also show sources
        print("\n📚 Relevant code sources found in RAG:")
        for i, doc in enumerate(rag_result["source_documents"][:3], 1):
            src = doc.metadata.get("source", "unknown")
            print(f"  {i}. {src}")

    combined_context = explicit_content + "\n\n" + rag_context
    return combined_context

# ========== COMMAND HANDLERS ==========
def handle_create(qa, user_input):
    """Create a new Swift file based on request, using project context."""
    request = user_input[7:].strip()
    if not request:
        print("Usage: create: <description of what to create>")
        return

    print(f"🔍 Analyzing request: {request}")
    # Retrieve relevant existing code from project
    context_code = retrieve_relevant_code(qa, request, PROJECT_DIR)

    prompt = f"""You are a Swift 6 expert helping a developer build an iOS/macOS app.
The developer already has the following code in their project (relevant parts):

{context_code}

Based on the above existing code (models, protocols, architecture patterns), generate a new Swift file that fulfills:
"{request}"

Requirements:
- Follow the exact naming conventions, coding style, and architecture visible in the existing code.
- If existing code uses actors, use actors. If it uses classes with completion handlers, follow that pattern unless Swift 6 concurrency is better.
- Use Swift 6 best practices (Sendable, async/await, actor isolation) when appropriate.
- The code must be complete, compilable, and well‑commented.
- Return ONLY the Swift code, no markdown, no explanations.

Generated Swift code:
"""
    response = qa.llm.invoke(prompt)

    # Clean up markdown if present
    code = response.strip()
    if code.startswith("```swift"):
        code = code[8:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    code = code.strip()

    print("\n🤖 Generated code:\n")
    print(code)
    print("\n" + "="*60)

    filename = input("Enter filename to save (e.g., ApiClient.swift): ").strip()
    if not filename.endswith(".swift"):
        filename += ".swift"
    
    # Save the newly created file in the root folder of the project directory (or custom path)
    filepath = os.path.join(PROJECT_DIR, filename)
    write_swift_file(filepath, code)
    print(f"✨ New file created at {filepath}")

def handle_improve(qa, user_input):
    """Improve an existing Swift file by applying Swift 6 best practices and project consistency recursively."""
    request = user_input[8:].strip()
    if not request:
        print("Usage: improve: <filename.swift> [additional hints]")
        return

    # Extract filename
    match = re.search(r'(\S+\.swift)', request)
    if not match:
        print("❌ Please specify a .swift file to improve (e.g., improve: NetworkManager.swift)")
        return
    filename = match.group(1)

    # DYNAMIC SEARCH: Look for the file recursively inside PROJECT_DIR
    filepath = None
    for root, _, files in os.walk(PROJECT_DIR):
        if filename in files:
            filepath = os.path.join(root, filename)
            break

    if not filepath:
        print(f"❌ File not found recursively inside: {PROJECT_DIR}")
        return

    print(f"🎯 Found local file path: {filepath}")
    original = read_swift_file(filepath)
    if original.startswith("Error"):
        print(original)
        return

    # Retrieve relevant project context (other files that interact with this one)
    context_prompt = f"Files that might relate to or use {filename}"
    context_code = retrieve_relevant_code(qa, context_prompt, PROJECT_DIR)

    improvement_prompt = f"""You are a Swift 6 expert. Below is an existing Swift file and other related files from the same project.

    Goal: Improve the file "{filename}" following Swift 6 best practices and making it consistent with the project's architecture.

    Original file:
    ```swift
    {original}
    ```

    Other relevant project code:
    {context_code}

    Improve the code by:
    - Adding proper actor isolation and Sendable conformance where needed.
    - Updating to async/await if currently using completion handlers.
    - Fixing any potential data races or concurrency issues.
    - Following naming conventions and patterns from the context.
    - Keeping the same public API unless it's unsafe or violates concurrency rules.

    Return the FULL improved file content (complete and ready to replace the original). No markdown, no explanations.
    """
    improved = qa.llm.invoke(improvement_prompt)

    improved = improved.strip()
    if improved.startswith("```swift"):
        improved = improved[8:]
    elif improved.startswith("```"):
        improved = improved[3:]
    if improved.endswith("```"):
        improved = improved[:-3]
    improved = improved.strip()

    print("\n🔍 Suggested improvements:\n")
    print(improved)
    print("\n" + "="*60)

    confirm = input("Apply these changes? (y/n): ").strip().lower()
    if confirm == 'y':
        backup = filepath + ".backup"
        write_swift_file(backup, original)
        write_swift_file(filepath, improved)
        print(f"✅ File updated. Backup saved as {backup}")
    else:
        print("❌ No changes applied.")

def handle_question(qa, user_input):
    """Normal teaching mode with dynamic file detection and auto-injection of source files."""
    print(f"🔍 Analyzing question...")
    
    # Auto-detect file references (e.g. NetworkManager.swift) in the text
    mentioned_files = re.findall(r'(\b\w+\.swift\b)', user_input)
    explicit_context = ""
    
    for fname in mentioned_files:
        # Locate the mentioned file recursively
        filepath = None
        for root, _, files in os.walk(PROJECT_DIR):
            if fname in files:
                filepath = os.path.join(root, fname)
                break
                
        if filepath and os.path.exists(filepath):
            content = read_swift_file(filepath)
            explicit_context += f"\n\n// ----- Local Context: {fname} -----\n{content}\n"
            print(f"📎 Auto-attached local file: {fname}")

    if explicit_context:
        enriched_query = f"""The user is asking a question about their project code.
Here is the source code of the file(s) they are asking about:
{explicit_context}

User Question: {user_input}"""
        response = qa.invoke(enriched_query)
    else:
        response = qa.invoke(user_input)
        
    print("\n🤖 Answer:")
    print(response["result"])
    print("\n📚 Sources used:")
    for i, doc in enumerate(response["source_documents"][:4], 1):
        src = doc.metadata.get("source", "unknown")
        print(f"  {i}. {src}")

def main():
    print("🧠 Swift AI Teacher with Project Awareness")
    print("Commands:")
    print(" - Ask any Swift coding question (uses your PDFs + project code)")
    print(" - create: <description> → generates a new .swift file consistent with your project")
    print(" - improve: <filename.swift> → improves an existing file using Swift 6 + project context")
    print(" - exit")
    print("-" * 60)

    qa = load_qa()
    if not qa:
        print("⚠️ No vector database. Please run: python3 ingest.py")
        return

    while True:
        user_input = input("\n❯ ").strip()
        if user_input.lower() == "exit":
            break

        if user_input.lower().startswith("create:"):
            handle_create(qa, user_input)
        elif user_input.lower().startswith("improve:"):
            handle_improve(qa, user_input)
        else:
            handle_question(qa, user_input)

if __name__ == "__main__":
    # Ensure project directory exists
    os.makedirs(PROJECT_DIR, exist_ok=True)
    main()