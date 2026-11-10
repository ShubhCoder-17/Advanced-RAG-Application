import os
import uuid
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
    
    def generate_embedding(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts)

class VectorStore:
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "data/vector_store"):
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        ids = [f"doc_{uuid.uuid4().hex[:8]}" for _ in range(len(documents))]
        self.collection.add(
            ids=ids,
            metadatas=[dict(doc.metadata) for doc in documents],
            documents=[doc.page_content for doc in documents],
            embeddings=embeddings.tolist()
        )

class RagRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_manager.generate_embedding([query])[0]
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        retrieved_docs = []
        if results['documents'] and results['documents'][0]:
            for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
                retrieved_docs.append({"document": doc, "metadata": meta, "score": 1 - dist})
        return retrieved_docs

def main():
    print("RAG System initialized.")
    # Example usage logic could go here
    pass

if __name__ == "__main__":
    main()
