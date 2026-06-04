from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
import os

# ---- Setup ----
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./chroma_ios_db",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # top 4 chunks

llm = ChatOllama(
    model="gemma4:latest",
    temperature=0.2,     # lower for factual answers
    num_predict=2048,
    repeat_penalty=1.1,
)

# ---- Prompt template (acts like a senior iOS mentor) ----
system_prompt = """
You are an expert iOS developer and mentor specializing in Swift, SwiftUI, actors (Swift concurrency),
best software architecture (MVVM, Clean Architecture, TCA), and WebSocket real‑time features.

Your mission is to help the user learn and build better software. You will either:
1. Teach and explain concepts, OR
2. Review and improve actual Swift code.

When the user shares a **code snippet** (any Swift code, no matter how small), do the following:
- Immediately recognise it as code.
- Directly critique it: point out potential issues (race conditions, retain cycles, UIKit/SwiftUI misuse, architectural weaknesses, etc.).
- Suggest concrete refactors with code examples.
- Base your critique on the official documentation context provided below.
- If the code is incomplete or unclear, ask one clarifying question before giving advice.

When the user asks a **conceptual question**, fall back to your teaching role:
- Explain from first principles.
- Use the provided documentation context.
- Give step‑by‑step examples.

Context from official Apple documentation and trusted sources:
{context}
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# Chain: retrieve → stuff docs → LLM
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ---- Session memory (keep a history per conversation) ----
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


session_id = "my-ios-learning-1"
print("💡 Tip: Type a Swift file path (e.g., ./MyCode.swift) to review a whole file.")

while True:
    user_input = input("\n🧑‍💻 You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # ---- FILE READING LOGIC ----
    input_text = user_input          # default: just the text typed
    file_path = None
    if user_input.startswith("file:"):
        file_path = user_input[5:].strip()
    elif os.path.isfile(user_input.strip()):
        file_path = user_input.strip()

    if file_path:
        try:
            with open(file_path, 'r') as f:
                code = f.read()
            input_text = (
                f"Please review this Swift code from file `{file_path}` "
                f"and suggest improvements:\n\n```swift\n{code}\n```"
            )
            print(f"📂 Loaded {file_path} ({len(code)} characters)")
        except Exception as e:
            print(f"❌ Could not read file: {e}")
            continue   # skip sending if file read fails

    # ---- Send to tutor ----
    try:
        result = conversational_rag_chain.invoke(
            {"input": input_text},
            config={"configurable": {"session_id": session_id}}
        )
        print("\n🧠 Tutor:", result["answer"])
    except Exception as e:
        print("\n❌ Error:", e)