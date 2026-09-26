"""
Admin Document Service for Demo 4.
Coordinates multi-format ingestion, chunking, ChromaDB updates, and BM25 index synchronization.
"""

from pathlib import Path
from typing import List, Dict, Any
import uuid
import shutil
from config import settings
from rag.loaders import extract_document
from rag.chunker import create_chunks
from rag.vector_store import vector_store_manager
from rag.keyword_search import bm25_search_manager


class DocumentService:
    def __init__(self):
        self.vector_store = vector_store_manager
        self.bm25 = bm25_search_manager
        self.sync_bm25_index()

    def sync_bm25_index(self):
        """Initializes or re-indexes BM25 from all current ChromaDB chunks."""
        all_chunks = self.vector_store.get_all_chunks()
        self.bm25.build_index(all_chunks)

    def ingest_file(self, temp_file_path: Path, original_filename: str) -> Dict[str, Any]:
        """
        Ingests an uploaded document into the hybrid knowledge base:
        1. Saves file permanently to uploads directory
        2. Extracts text & metadata
        3. Splits into chunks
        4. Embeds & indexes into ChromaDB
        5. Syncs BM25 search index
        """
        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        
        # Save to uploads directory
        dest_filename = f"{doc_id}_{original_filename}"
        dest_path = settings.absolute_upload_dir / dest_filename
        shutil.copyfile(temp_file_path, dest_path)

        # 1. Extract documents
        raw_docs = extract_document(dest_path, doc_id=doc_id)
        if not raw_docs:
            raise ValueError(f"Could not extract readable text from '{original_filename}'.")

        # Override source metadata with clean original filename
        for doc in raw_docs:
            doc.metadata["source"] = original_filename
            doc.metadata["file_path"] = str(dest_path)

        # 2. Chunk text
        chunks = create_chunks(raw_docs)

        # 3. Add to ChromaDB vector store
        added_count = self.vector_store.add_chunks(chunks)

        # 4. Refresh BM25 index with newly added documents
        self.sync_bm25_index()

        # Compute page count
        pages = set(c.metadata.get("page", 1) for c in chunks)

        return {
            "doc_id": doc_id,
            "filename": original_filename,
            "format": dest_path.suffix.lstrip(".").lower(),
            "chunks_count": added_count,
            "pages_count": len(pages),
            "status": "indexed",
        }

    def ingest_url(self, url: str) -> Dict[str, Any]:
        """
        Safely fetches, sanitizes, and indexes a website webpage:
        1. Validates SSRF and safe IP resolution
        2. Strips scripts, styles, malware vectors, and hidden prompt injections
        3. Splits into semantic chunks
        4. Indexes into ChromaDB vector store
        5. Syncs BM25 lexical search index
        """
        from rag.loaders import load_url

        doc_id = f"web_{uuid.uuid4().hex[:12]}"

        # 1. Fetch & sanitize web content
        raw_docs = load_url(url, doc_id=doc_id)
        if not raw_docs:
            raise ValueError(f"No usable content could be extracted from '{url}'.")

        title = raw_docs[0].metadata.get("source", url)

        # 2. Chunk text
        chunks = create_chunks(raw_docs)

        # 3. Add to ChromaDB
        added_count = self.vector_store.add_chunks(chunks)

        # 4. Refresh BM25
        self.sync_bm25_index()

        return {
            "doc_id": doc_id,
            "filename": title,
            "format": "web",
            "chunks_count": added_count,
            "pages_count": 1,
            "status": "indexed",
        }


    def list_documents(self) -> List[Dict[str, Any]]:
        """Returns the list of all currently indexed documents."""
        return self.vector_store.list_indexed_documents()

    def delete_document(self, doc_id: str) -> bool:
        """Deletes a document from vector store, removes file, and refreshes BM25."""
        deleted_count = self.vector_store.delete_document(doc_id)
        
        # Re-sync BM25 index after deletion
        self.sync_bm25_index()

        # Clean up files matching doc_id in uploads
        for f in settings.absolute_upload_dir.glob(f"{doc_id}_*"):
            try:
                f.unlink()
            except Exception:
                pass

        return deleted_count > 0


# Global singleton instance
document_service = DocumentService()
