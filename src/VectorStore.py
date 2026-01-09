import os
from typing import List, Any
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from uuid import uuid4


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

        self.client = chromadb.Client(
            Settings(
                persist_directory=persist_dir,
                anonymized_telemetry=False
            )
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

        print(f"[INFO] VectorStore initialized")
        print(f"       Model: {embedding_model}")
        print(f"       Persist dir: {persist_dir}")

    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        "Add documents to the vector store"
        if not self.collection:
            raise ValueError("Collection not initialized")
        if len(documents) != embeddings.shape[0]:
            raise ValueError("Number of documents and embeddings do not match")
        print(f"Adding {len(documents)} documents to the vector store ...")
        ids = []
        metadatas = []
        documents_list = []
        embedding_lists = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            documents_list.append(doc.page_content)
            embedding_lists.append(embedding.tolist())
        try:
            self.collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=documents_list,
                embeddings=embedding_lists
            )
            print(f"Successfully added {len(documents)} documents to the vector store")
            print(f"Collection {self.collection} and num collections {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents to the vector store: {e}")
            raise
    def query(
        self,
        query_text: str,
        top_k: int = 5
    ):
        print(f"[INFO] Query: {query_text}")

        query_embedding = self.embedding_model.encode([query_text]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances", "ids"]
        )

        return results
