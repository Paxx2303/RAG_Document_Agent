from src.VectorStore import VectorStore
from src.DataLoader import load_all_documents
from src.Retrieval import RAG_Retrieval

if __name__ == "__main__":
    documents = load_all_documents('./data')
    vectocstore = VectorStore("chroma_store")
    rag = RAG_Retrieval()
    query = "what is streamlit"
    summary = rag.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
