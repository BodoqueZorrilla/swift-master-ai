from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import sys

# Initialize components
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(
    model="gemma4:12b",
    temperature=0.2,        # Lower = more focused, good for code review
    num_ctx=8192,           # Larger context for code + retrieved docs
    base_url="http://localhost:11434"
)

# Load existing vector store
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="swift_knowledge"
)

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.3}
)

# Custom prompt optimized for Swift 6 / SwiftUI mentorship
SWIFT_MENTOR_PROMPT = PromptTemplate.from_template("""
You are a senior iOS developer and SwiftUI architect with 10+ years of experience.
You specialize in Swift 6, SwiftUI, and modern Apple frameworks.

CONTEXT FROM YOUR KNOWLEDGE BASE:
{context}

USER'S CODE / QUESTION:
{question}

INSTRUCTIONS:
- Analyze the code using Swift 6 best practices (strict concurrency, actors, @Observable, typed throws)
- Reference specific patterns from the provided context when relevant
- Suggest improvements with concrete code examples
- Flag potential data races, memory leaks, or Swift 6 migration issues
- Keep explanations concise but technically deep
- If the context doesn't cover the topic, rely on your training knowledge

RESPONSE:
""")

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": SWIFT_MENTOR_PROMPT}
)

def chat():
    print("🍎 Swift 6 Mentor (Gemma 4 12B) — Type 'quit' to exit")
    print("=" * 60)
    
    while True:
        try:
            query = input("\n💻 Your code/question:\n> ")
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            print("\n🤖 Thinking...")
            result = qa_chain.invoke({"query": query})
            
            print(f"\n📋 ANSWER:\n{result['result']}")
            
            # Show which PDFs were referenced
            sources = set([
                doc.metadata.get('source', 'unknown').split('/')[-1] 
                for doc in result['source_documents']
            ])
            if sources:
                print(f"\n📚 Sources: {', '.join(sources)}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n👋 Happy coding!")

if __name__ == "__main__":
    chat()