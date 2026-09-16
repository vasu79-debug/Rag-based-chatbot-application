"""
Smart Text Chunker for Demo 4 RAG.
Uses RecursiveCharacterTextSplitter to split documents while preserving metadata.
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(
    documents: List[Document],
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> List[Document]:
    """
    Splits documents into overlapping chunks with preserved metadata.
    """
    if not documents:
        return []

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
        chunk.metadata["chunk_id"] = f"{doc_id}_chunk_{i}"
        chunk.metadata["chunk_index"] = i

    return chunks
