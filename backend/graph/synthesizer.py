"""
Synthesizer Node for Demo 4 LangGraph.
Merges knowledge streams into clearly labeled, dual-structured sections with citations.
"""

from typing import Dict, Any
from graph.state import AgentState


def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """Combines RAG and General answers into a structured dual-knowledge response."""
    route = state.get("route", "HYBRID")
    rag_answer = state.get("rag_answer")
    rag_available = state.get("rag_available", False)
    citations = state.get("citations", [])
    general_answer = state.get("general_answer")

    org_section = None
    general_section = None

    if route == "RAG" or (route == "HYBRID" and rag_available and rag_answer):
        org_section = {
            "label": "🏢 Organisational Knowledge (Grounded in Your Files)",
            "content": rag_answer,
            "citations": citations,
            "has_citations": len(citations) > 0,
        }

    if route == "GENERAL" or route == "HYBRID":
        if general_answer:
            general_section = {
                "label": "🌐 General Knowledge & AI Intelligence",
                "content": general_answer,
            }

    # If in RAG mode only, but no RAG chunks were available, fallback to general advice
    if route == "RAG" and not rag_available and general_answer:
        general_section = {
            "label": "🌐 General Knowledge & Guidance",
            "content": general_answer,
        }

    final_payload = {
        "route": route,
        "org_section": org_section,
        "general_section": general_section,
        "citations_count": len(citations),
        "citations": citations,
    }

    return {
        "final_output": final_payload,
    }
