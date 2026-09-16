"""
Persistent ChromaDB Vector Store for Demo 4.
Handles embeddings, vector search, document ingestion, and deletion.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from config import settings


class VectorStoreManager:
    def __init__(self):
        self.persist_dir = str(settings.absolute_chroma_dir)
        self.collection_name = "demo4_knowledge_base"
        
        # Initialize local embeddings (runs locally, zero cost)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Persistent Chroma client
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[Document]) -> int:
        """Adds text chunks with embeddings and metadata into ChromaDB."""
        if not chunks:
            return 0

        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        ids = [c.metadata.get("chunk_id", str(i)) for i, c in enumerate(chunks)]

        # Generate embeddings
        embeddings_list = self.embeddings.embed_documents(texts)

        # Upsert into ChromaDB
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings_list,
            documents=texts,
            metadatas=metadatas,
        )
        return len(chunks)

    def similarity_search(self, query: str, k: int = 20) -> List[Document]:
        """Performs cosine similarity search against ChromaDB."""
        if self.collection.count() == 0:
            return []

        query_embedding = self.embeddings.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        documents = []
        if results and results.get("documents") and len(results["documents"]) > 0:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]

            for doc_text, meta, dist in zip(docs, metas, distances):
                # Cosine distance to similarity: similarity = 1 - distance
                similarity_score = max(0.0, 1.0 - float(dist))
                meta_copy = dict(meta) if meta else {}
                meta_copy["score"] = round(similarity_score, 4)
                documents.append(Document(page_content=doc_text, metadata=meta_copy))

        return documents

    def delete_document(self, doc_id: str) -> int:
        """Deletes all chunks belonging to a specific doc_id."""
        if self.collection.count() == 0:
            return 0

        # Query chunks with matching doc_id
        res = self.collection.get(
            where={"doc_id": doc_id},
            include=["metadatas"],
        )
        ids_to_delete = res.get("ids", [])
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    def get_all_chunks(self) -> List[Document]:
        """Retrieves all stored chunks (used to build/sync the BM25 index)."""
        if self.collection.count() == 0:
            return []

        res = self.collection.get(include=["documents", "metadatas"])
        docs = []
        for text, meta in zip(res.get("documents", []), res.get("metadatas", [])):
            docs.append(Document(page_content=text, metadata=meta or {}))
        return docs

    def list_indexed_documents(self) -> List[Dict[str, Any]]:
        """Aggregates all indexed documents with chunk statistics."""
        chunks = self.get_all_chunks()
        doc_map = {}

        for c in chunks:
            doc_id = c.metadata.get("doc_id", "unknown")
            source = c.metadata.get("source", "untitled")
            fmt = c.metadata.get("format", "unknown")

            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "doc_id": doc_id,
                    "filename": source,
                    "format": fmt,
                    "chunks_count": 0,
                    "pages_count": set(),
                }
            doc_map[doc_id]["chunks_count"] += 1
            if "page" in c.metadata:
                doc_map[doc_id]["pages_count"].add(c.metadata["page"])

        summary = []
        for d in doc_map.values():
            summary.append({
                "doc_id": d["doc_id"],
                "filename": d["filename"],
                "format": d["format"],
                "chunks_count": d["chunks_count"],
                "pages_count": len(d["pages_count"]),
            })
        return summary


# Global singleton instance
vector_store_manager = VectorStoreManager()
