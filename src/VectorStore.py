"""
VectorStore.py - Fixed Methods
Sửa lại add_documents và build_from_documents
"""

import os
from typing import List, Any
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from uuid import uuid4
from src.Embedding import Embedding


class VectorStore:
    def __init__(
            self,
            persist_dir: str = "./vector_db",
            collection_name: str = "rag_docs",
            embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        self.embedding_model_name = embedding_model
        self.embedding_model = SentenceTransformer(embedding_model)

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                anonymized_telemetry=False
            )
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

        print(f"[INFO] VectorStore initialized")
        print(f"       Model: {embedding_model}")
        print(f"       Persist dir: {persist_dir}")
        print(f"       Current documents: {self.collection.count()}")

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Add documents to the vector store

        Args:
            documents: List of LangChain Document objects
            embeddings: numpy array of shape (n_docs, embedding_dim)
        """
        if not self.collection:
            raise ValueError("Collection not initialized")

        if len(documents) != embeddings.shape[0]:
            raise ValueError(
                f"Number of documents ({len(documents)}) and embeddings "
                f"({embeddings.shape[0]}) do not match"
            )

        print(f"[INFO] Adding {len(documents)} documents to vector store...")

        ids = []
        metadatas = []
        documents_list = []
        embedding_lists = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique ID
            doc_id = f"doc_{uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            # Prepare metadata
            metadata = dict(doc.metadata) if hasattr(doc, 'metadata') else {}
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            # Add document text
            documents_list.append(doc.page_content)

            # Convert embedding to list
            embedding_lists.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=documents_list,
                embeddings=embedding_lists
            )
            print(f"[SUCCESS] Added {len(documents)} documents")
            print(f"[INFO] Total documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"[ERROR] Failed to add documents: {e}")
            raise

    def build_from_documents(self, documents: List[Any]):
        """
        Build vector store from raw documents
        Tự động: split -> embed -> add

        Args:
            documents: List of LangChain Document objects (raw, chưa split)
        """
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")

        # Step 1: Split documents into chunks
        print("[1/3] Splitting documents into chunks...")
        chunks = Embedding.split_doc_to_chunk(
            documents,
            chunks_size=1000,
            chunk_overlap=200
        )
        print(f"[INFO] Created {len(chunks)} chunks")

        # Step 2: Generate embeddings
        print("[2/3] Generating embeddings...")
        emb_model = Embedding(model_name=self.embedding_model_name)
        embeddings = emb_model.embed_chunks(chunks)
        print(f"[INFO] Generated embeddings shape: {embeddings.shape}")

        # Step 3: Add to vector store
        print("[3/3] Adding to vector store...")
        self.add_documents(
            documents=chunks,  # ✅ Pass chunks (List[Document])
            embeddings=embeddings  # ✅ Pass embeddings (np.ndarray)
        )

        print(f"[SUCCESS] Vector store built successfully!")
        print(f"[INFO] Persisted to: {self.persist_dir}")

    def query(
            self,
            query_text: str,
            top_k: int = 5
    ):
        """
        Query vector store for similar documents

        Args:
            query_text: Query string
            top_k: Number of results to return

        Returns:
            Dictionary with 'documents', 'metadatas', 'distances'
        """
        print(f"[INFO] Querying: {query_text}")

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query_text]).tolist()

        # Search in collection
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        print(f"[INFO] Found {len(results['documents'][0])} results")
        return results

    def clear(self):
        """Clear all documents from collection"""
        print(f"[WARNING] Clearing collection...")
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(self.collection.name)
        print(f"[INFO] Collection cleared")

    def get_stats(self):
        """Get collection statistics"""
        count = self.collection.count()
        stats = {
            'total_documents': count,
            'collection_name': self.collection.name,
            'persist_dir': self.persist_dir,
            'embedding_model': self.embedding_model_name
        }
        return stats


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    from src.DataLoader import load_all_documents

    print("=" * 60)
    print("🧪 TEST VECTORSTORE")
    print("=" * 60)

    # Initialize
    vectorstore = VectorStore(
        persist_dir="./test_vector_db",
        collection_name="test_docs"
    )

    # Check existing data
    stats = vectorstore.get_stats()
    print(f"\n📊 Stats: {stats}")

    if stats['total_documents'] == 0:
        print("\n🔨 Building vector store...")

        # Load documents
        documents = load_all_documents('./data')

        if documents:
            # Build (auto: split -> embed -> add)
            vectorstore.build_from_documents(documents)
        else:
            print("⚠️  No documents found in ./data")
    else:
        print("✅ Vector store already has data")

    # Test query
    print("\n" + "=" * 60)
    print("🔍 TEST QUERY")
    print("=" * 60)

    results = vectorstore.query("what is langchain", top_k=3)

    print("\nResults:")
    for i, (doc, meta, dist) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
    )):
        print(f"\n--- Result {i + 1} ---")
        print(f"Distance: {dist:.4f}")
        print(f"Source: {meta.get('source', 'N/A')}")
        print(f"Content: {doc[:200]}...")

    # Final stats
    print("\n" + "=" * 60)
    print("📊 FINAL STATS")
    print("=" * 60)
    stats = vectorstore.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
