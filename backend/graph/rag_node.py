"""
Organisational Knowledge (RAG) Node for Demo 4 LangGraph.
Executes Hybrid Search + Reranking and generates grounded company responses with citations.
"""

from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage
from graph.state import AgentState, Citation
from graph.llm_factory import get_chat_model
from rag.hybrid_retriever import hybrid_retriever


RAG_PROMPT = """You are the Organisational Knowledge specialist for this enterprise assistant.
Your task is to answer the query using ONLY the provided company document excerpts below.

RULES:
1. Answer factually, concisely, and strictly based on the provided context.
2. Quote figures, numbers, dates, and names exactly as stated in the documents.
3. If the context does not contain enough information to answer, state clearly: "The internal documents do not contain specific information regarding this query."
4. Do not include external world knowledge here (that will be provided separately).
5. Output clean, direct markdown without generic conversational filler.

CONTEXT DOCUMENTS:
{context}
"""


def rag_node(state: AgentState) -> Dict[str, Any]:
    """Retrieves company documents and generates the organizational knowledge stream."""
    sub_questions = state.get("sub_questions", {})
    query = sub_questions.get("rag") or state["question"]

    # 1. Hybrid Retrieval + Rerank
    retrieved_chunks = hybrid_retriever.retrieve(query)

    if not retrieved_chunks:
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

    # 3. Generate grounded organizational response
    llm = get_chat_model(temperature=0.1, max_tokens=800)
    prompt = RAG_PROMPT.format(context=context_str)

    try:
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"Question: {query}"),
        ])
        rag_answer = response.content.strip()
    except Exception as e:
        rag_answer = f"Error generating internal document response: {str(e)}"

    return {
        "rag_chunks": retrieved_chunks,
        "citations": citations,
        "rag_answer": rag_answer,
        "rag_available": True,
    }
