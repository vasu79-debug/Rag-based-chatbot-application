import logging
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage
from graph.state import AgentState, Citation
from graph.llm_factory import get_chat_model
from rag.hybrid_retriever import hybrid_retriever

logger = logging.getLogger("demo4.graph.rag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


from config import settings


def build_rag_prompt(context_str: str) -> str:
    """Builds the RAG prompt dynamically using company settings."""
    company_name = getattr(settings, "COMPANY_NAME", "the organization")
    tone = getattr(settings, "ASSISTANT_TONE", "factual, concise, and structured")
    custom_rules = getattr(settings, "CUSTOM_SYSTEM_INSTRUCTIONS", "")

    return f"""You are the Organisational Knowledge specialist for {company_name}.
Your task is to answer the query using ONLY the provided company document excerpts below.

RULES:
1. Tone: {tone}.
2. Answer factually and strictly based on the provided context.
3. Quote figures, numbers, dates, and names exactly as stated in the documents.
4. If the context does not contain enough information to answer, state clearly that the internal documents do not contain specific information regarding this query, or highlight the nearest relevant information found.
5. Do not include external world knowledge here (that will be provided separately).
6. Output clean, direct markdown without generic conversational filler.
7. {custom_rules}

CONTEXT DOCUMENTS:
{context_str}
"""



def rag_node(state: AgentState) -> Dict[str, Any]:
    """Retrieves company documents and generates the organizational knowledge stream with step logs."""
    sub_questions = state.get("sub_questions", {})
    query = sub_questions.get("rag") or state["question"]

    logger.info("━" * 60)
    logger.info(f"🏢 [Step 2/4: RAG STREAM] Running Retrieval on Internal Knowledge Base")
    logger.info(f"   Query: \"{query}\"")

    # 1. Hybrid Retrieval + Rerank
    retrieved_chunks = hybrid_retriever.retrieve(query)

    if not retrieved_chunks:
        logger.info("   ⚠️  [RAG Stream] No matching internal documents found in vector/lexical index.")
        logger.info("━" * 60)
        return {
            "rag_chunks": [],
            "citations": [],
            "rag_answer": "No matching internal documents were found for this query.",
            "rag_available": False,
        }

    # 2. Extract structured citations
    citations: List[Citation] = []
    context_blocks = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        source = chunk.metadata.get("source", "Document")
        page = chunk.metadata.get("page", 1)
        fmt = chunk.metadata.get("format", "doc")
        score = chunk.metadata.get("rerank_score", chunk.metadata.get("score", 0.0))
        snippet = chunk.page_content[:200] + ("…" if len(chunk.page_content) > 200 else "")

        citations.append({
            "source": source,
            "page": page,
            "format": fmt,
            "snippet": snippet,
            "score": score,
        })

        context_blocks.append(
            f"--- SOURCE [{i}]: {source} (Page {page}) ---\n{chunk.page_content}"
        )

    context_str = "\n\n".join(context_blocks)
    logger.info(f"   📑 [RAG Stream] Grounding LLM with {len(citations)} verified citations:")
    for cit in citations:
        logger.info(f"      • {cit['source']} (Page {cit['page']}) - Relevance: {round(cit['score']*100, 1)}%")

    # 3. Generate grounded organizational response
    llm = get_chat_model(temperature=0.1, max_tokens=1000)
    prompt = build_rag_prompt(context_str)


    try:
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"Question: {query}"),
        ])
        rag_answer = response.content.strip() if response and response.content else ""
        if not rag_answer and retrieved_chunks:
            # Resilient fallback from top chunk content if token limit was exceeded
            rag_answer = "\n\n".join([c.page_content for c in retrieved_chunks[:2]])
        logger.info(f"   ✅ [RAG Stream] Grounded Answer Generated ({len(rag_answer)} chars)")
        logger.info(f"   [RAG Output]: {rag_answer}")
    except Exception as e:
        logger.error(f"   ❌ [RAG Stream] Error generating RAG response: {str(e)}")
        rag_answer = f"Error generating internal document response: {str(e)}"

    logger.info("━" * 60)

    return {
        "rag_chunks": retrieved_chunks,
        "citations": citations,
        "rag_answer": rag_answer,
        "rag_available": True,
    }

