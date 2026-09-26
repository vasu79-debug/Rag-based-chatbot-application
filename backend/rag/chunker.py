"""
Smart Text Chunker for Demo 4 RAG.
Uses RecursiveCharacterTextSplitter to split documents while preserving metadata.
"""

import logging
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("demo4.rag.chunker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def create_chunks(
    documents: List[Document],
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> List[Document]:
    """
    Splits documents into overlapping chunks with preserved metadata and logs chunking details.
    """
    if not documents:
        logger.warning("[Chunker] No documents provided to chunk.")
        return []

    source_name = documents[0].metadata.get("source", "Document")
    total_raw_chars = sum(len(d.page_content) for d in documents)
    logger.info("━" * 60)
    logger.info(f"✂️  [Chunker] Splitting '{source_name}' ({len(documents)} sections/pages, {total_raw_chars} total characters)")
    logger.info(f"    Settings: Chunk Size={chunk_size} chars | Overlap={chunk_overlap} chars")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)

    # Enrich each chunk with a unique chunk_id and index
    for i, chunk in enumerate(chunks):
        doc_id = chunk.metadata.get("doc_id", "doc")
        chunk_id = f"{doc_id}_chunk_{i}"
        chunk.metadata["chunk_id"] = chunk_id
        chunk.metadata["chunk_index"] = i
        
        page = chunk.metadata.get("page", 1)
        char_len = len(chunk.page_content)
        snippet_preview = chunk.page_content[:60].replace("\n", " ") + ("..." if char_len > 60 else "")
        
        logger.info(f"   ├─ [Chunk #{i+1}/{len(chunks)}] ID: {chunk_id} | Page: {page} | Size: {char_len} chars | Preview: \"{snippet_preview}\"")

    logger.info(f"✅ [Chunker] Successfully generated {len(chunks)} chunks for '{source_name}'")
    logger.info("━" * 60)

    return chunks

