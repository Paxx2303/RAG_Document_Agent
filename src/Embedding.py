from typing import *
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
import numpy as np
from src.DataLoader import load_all_documents

from sentence_transformers import SentenceTransformer
class Embedding:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def split_doc_to_chunk(docs , chunks_size = 1000 , chunk_overlap = 20):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunks_size,
            chunk_overlap = chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(docs)
        return chunks
    def embed_chunks(self, chunks):
        texts = [chunk.page_content for chunk in chunks]
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings
