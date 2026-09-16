"""
Unified Hybrid Retriever with Reciprocal Rank Fusion (RRF) & Cross-Encoder Reranking.
Combines ChromaDB vector search + BM25 keyword search + FlashRank reranker.
"""

from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from config import settings
from rag.vector_store import vector_store_manager
from rag.keyword_search import bm25_search_manager
from rag.reranker import reranker_manager


def reciprocal_rank_fusion(
    vector_docs: List[Document],
    bm25_docs: List[Document],
    rrf_k: int = 60,
) -> List[Document]:
    """
    Applies Reciprocal Rank Fusion (RRF) algorithm to merge two ranked candidate lists:
    RRF_Score(d) = sum(1 / (rrf_k + rank(d)))
    """
    doc_map: Dict[str, Document] = {}
    scores: Dict[str, float] = {}

    # Rank vector search results
    for rank, doc in enumerate(vector_docs):
        chunk_id = doc.metadata.get("chunk_id", doc.page_content[:40])
        doc_map[chunk_id] = doc
        scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank + 1))

    # Rank BM25 results
    for rank, doc in enumerate(bm25_docs):
        chunk_id = doc.metadata.get("chunk_id", doc.page_content[:40])
        if chunk_id not in doc_map:
            doc_map[chunk_id] = doc
        scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank + 1))

    # Sort merged documents by combined RRF score descending
    sorted_chunk_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    
    fused_docs = []
    for cid in sorted_chunk_ids:
        doc = doc_map[cid]
        doc.metadata["rrf_score"] = round(scores[cid], 5)
        fused_docs.append(doc)

    return fused_docs


class HybridRetriever:
    def __init__(self):
        self.vector_store = vector_store_manager
        self.bm25 = bm25_search_manager
        self.reranker = reranker_manager

    def retrieve(
        self,
        query: str,
        top_k_vector: int = None,
        top_k_bm25: int = None,
        top_n_rerank: int = None,
        threshold: float = None,
    ) -> List[Document]:
        """
        Executes the full hybrid pipeline:
        1. Dense Vector Search (ChromaDB)
        2. Sparse Lexical Search (BM25)
        3. Reciprocal Rank Fusion (RRF)
        4. Cross-Encoder Reranker
        5. Score Thresholding
        """
        k_vec = top_k_vector or settings.RAG_TOP_K_VECTOR
        k_bm25 = top_k_bm25 or settings.RAG_TOP_K_BM25
        n_rerank = top_n_rerank or settings.RAG_TOP_N_RERANK
        min_thresh = threshold if threshold is not None else settings.RAG_SIMILARITY_THRESHOLD

        # 1. Parallel / Sequential Retrieval from both stores
        vec_docs = self.vector_store.similarity_search(query, k=k_vec)
        bm25_docs = self.bm25.search(query, k=k_bm25)

        # 2. Fuse candidate pools with RRF
        fused_candidates = reciprocal_rank_fusion(vec_docs, bm25_docs)

        if not fused_candidates:
            return []

        # 3. Neural Reranking
        reranked_docs = self.reranker.rerank(query, fused_candidates, top_n=n_rerank)

        # 4. Filter by confidence threshold if rerank score exists
        filtered_docs = []
        for doc in reranked_docs:
            score = doc.metadata.get("rerank_score", doc.metadata.get("score", 1.0))
            if score >= min_thresh:
                filtered_docs.append(doc)

        return filtered_docs if filtered_docs else reranked_docs[:2]


# Global singleton instance
hybrid_retriever = HybridRetriever()
