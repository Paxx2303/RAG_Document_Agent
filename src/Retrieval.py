import os
from dotenv import load_dotenv
from src.VectorStore import  *
from langchain_groq import ChatGroq
import chromadb
load_dotenv()

class RAG_Retrieval:
    def __init__(self, persist_dir: str = "./vector_db", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama3-8b-8192"):
        self.vectorstore =VectorStore(persist_dir, embedding_model)
        client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=chromadb.Settings(
                anonymized_telemetry=False
            )
        )
        groq_api_key = ""
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke([prompt])
        return response.content

# Example usage
if __name__ == "__main__":
    rag_search = RAG_Retrieval()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)