import os
from dotenv import load_dotenv
from src.VectorStore import  *
from langchain_groq import ChatGroq
import chromadb
load_dotenv()

class RAG_Retrieval:
    def __init__(self, persist_dir: str = "./vector_db", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama3-8b-8192"):
        self.vectorstore =VectorStore(persist_dir, embedding_model)
        groq_api_key = ""
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        texts = [
            doc
            for doc, meta in zip(docs, metas)
            if meta is not None
        ]
        if not texts:
            return "No relevant documents found."
        context = "\n\n".join(texts)
        prompt = f"""You are a technical documentation assistant.
    Answer concisely using ONLY the context below.
    Context:
    {context}
    Question:
    {query}
    Answer:
    """
        response = self.llm.invoke(prompt)
        return response.content
# Example usage
if __name__ == "__main__":
    rag_search = RAG_Retrieval()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)