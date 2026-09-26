import logging
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from config import settings
from rag.vector_store import vector_store_manager
from rag.keyword_search import bm25_search_manager
from rag.reranker import reranker_manager

logger = logging.getLogger("demo4.rag.retriever")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


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
        Executes the full hybrid pipeline with comprehensive logging:
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

        logger.info("  🔍 [Hybrid Retriever] Querying knowledge base...")
        logger.info(f"     Search Target: \"{query}\"")

        # 1. Parallel / Sequential Retrieval from both stores
        vec_docs = self.vector_store.similarity_search(query, k=k_vec)
        bm25_docs = self.bm25.search(query, k=k_bm25)

        logger.info(f"     ├─ [ChromaDB Vector] Found {len(vec_docs)} semantic candidates (Top score: {vec_docs[0].metadata.get('score', 0) if vec_docs else 0.0})")
        logger.info(f"     ├─ [BM25 Keyword]    Found {len(bm25_docs)} lexical candidates (Top score: {round(bm25_docs[0].metadata.get('bm25_score', 0), 2) if bm25_docs else 0.0})")

        # 2. Fuse candidate pools with RRF
        fused_candidates = reciprocal_rank_fusion(vec_docs, bm25_docs)

        if not fused_candidates:
            logger.warning("     └─ ❌ [Hybrid Retriever] No matching candidates found in any index.")
            return []

        logger.info(f"     ├─ [RRF Fusion] Merged {len(fused_candidates)} unique candidates from vector + lexical pools")

        # 3. Neural Reranking
        reranked_docs = self.reranker.rerank(query, fused_candidates, top_n=n_rerank)
        logger.info(f"     ├─ [FlashRank Rerank] Re-scored top {len(reranked_docs)} passages with cross-encoder:")
        for idx, rdoc in enumerate(reranked_docs, 1):
            source = rdoc.metadata.get("source", "doc")
            cid = rdoc.metadata.get("chunk_id", "chunk")
            score = rdoc.metadata.get("rerank_score", 0.0)
            logger.info(f"     │  [{idx}] Score: {round(score*100, 1)}% | ID: {cid} | Source: {source}")

        # 4. Filter by confidence threshold if rerank score exists
        filtered_docs = []
        for doc in reranked_docs:
            score = doc.metadata.get("rerank_score", doc.metadata.get("score", 1.0))
            if score >= min_thresh:
                filtered_docs.append(doc)

        final_docs = filtered_docs if filtered_docs else reranked_docs[:2]
        logger.info(f"     └─ ✅ [Hybrid Retriever] Selected {len(final_docs)} final context passages (Threshold: {min_thresh})")

        return final_docs


# Global singleton instance
hybrid_retriever = HybridRetriever()

