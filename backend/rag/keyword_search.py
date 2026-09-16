"""
BM25 Lexical Keyword Search for Demo 4.
Provides exact keyword matching for contract codes, numbers, policy names, and acronyms.
"""

from typing import List
import re
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


def tokenize(text: str) -> List[str]:
    """Tokenizes text for BM25 keyword matching."""
    # Lowercase, clean punctuation, split on whitespace and alphanumeric tokens
    tokens = re.findall(r"\b\w+\b", text.lower())
    return tokens


class BM25SearchManager:
    def __init__(self):
        self.corpus_chunks: List[Document] = []
        self.bm25: BM25Okapi = None

    def build_index(self, chunks: List[Document]):
        """Builds or refreshes the BM25 index from all active chunks."""
        self.corpus_chunks = chunks
        if not chunks:
            self.bm25 = None
            return

        tokenized_corpus = [tokenize(doc.page_content) for doc in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, k: int = 20) -> List[Document]:
        """Performs lexical BM25 search and returns ranked candidate documents."""
        if not self.bm25 or not self.corpus_chunks:
            return []

        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []

        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Pair documents with their BM25 score
        scored_docs = []
        for doc, score in zip(self.corpus_chunks, doc_scores):
            if score > 0:
                doc_copy = Document(
                    page_content=doc.page_content,
                    metadata=dict(doc.metadata),
                )
                doc_copy.metadata["bm25_score"] = float(score)
                scored_docs.append((doc_copy, float(score)))

        # Sort by BM25 score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:k]]


# Global singleton instance
bm25_search_manager = BM25SearchManager()
