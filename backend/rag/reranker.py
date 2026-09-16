"""
Cross-Encoder / FlashRank Neural Reranker for Demo 4.
Re-scores candidate passages against the query for high-precision retrieval.
"""

from typing import List
from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document
from config import settings


class RerankerManager:
    def __init__(self):
        # FlashRank uses a local quantized cross-encoder (nano/mini, ultra-fast on CPU)
        try:
            self.ranker = Ranker(model_name=settings.RERANKER_MODEL, cache_dir="./data/models")
        except Exception:
            # Fallback to default lightweight ranker
            self.ranker = Ranker(cache_dir="./data/models")

    def rerank(self, query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
        """
        Takes candidate documents, re-ranks them using neural cross-attention,
        and returns the top_n most relevant chunks.
        """
        if not documents:
            return []

        # Prepare FlashRank input format
        passages = []
        for i, doc in enumerate(documents):
            passages.append({
                "id": doc.metadata.get("chunk_id", str(i)),
                "text": doc.page_content,
                "meta": doc.metadata,
            })

        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerank_request)

        reranked_docs = []
        for item in results[:top_n]:
            meta = item.get("meta", {})
            meta["rerank_score"] = round(float(item.get("score", 0.0)), 4)
            reranked_docs.append(
                Document(
                    page_content=item.get("text", ""),
                    metadata=meta,
                )
            )

        return reranked_docs


# Global singleton instance
reranker_manager = RerankerManager()
