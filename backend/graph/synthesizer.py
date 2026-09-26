import logging
from typing import Dict, Any
from graph.state import AgentState

logger = logging.getLogger("demo4.graph.synthesizer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """Combines RAG and General answers into a structured dual-knowledge response and logs the result."""
    route = state.get("route", "HYBRID")
    rag_answer = state.get("rag_answer")
    rag_available = state.get("rag_available", False)
    citations = state.get("citations", [])
    general_answer = state.get("general_answer")

    org_section = None
    general_section = None

    if route == "RAG" or (route == "HYBRID" and rag_available and rag_answer):
        org_section = {
            "label": "Organisational Knowledge (Grounded in Your Files)",
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

    # Log clearly which stream was retrieved and delivered
    logger.info("━" * 60)
    logger.info("📦 [Step 4/4: SYNTHESIZER] Response Summary:")
    if org_section and general_section:
        logger.info("   ⚡ RESULT TYPE: [HYBRID DUAL-STREAM] (Both RAG Docs & General LLM Reasoning delivered)")
        logger.info(f"      ├─ 🏢 Organisational Section: Present ({len(citations)} verified citations)")
        logger.info(f"      └─ 🌐 General LLM Section:    Present ({len(general_answer)} chars)")
    elif org_section:
        logger.info("   🏢 RESULT TYPE: [PURE RAG KNOWLEDGE] (100% Grounded in Uploaded Documents)")
        logger.info(f"      └─ Citations Attached: {len(citations)}")
    elif general_section:
        logger.info("   🌐 RESULT TYPE: [PURE GENERAL LLM] (World Knowledge / Reasoning / Code / Actions)")
        logger.info(f"      └─ Content Size: {len(general_answer)} chars")
    else:
        logger.info("   ⚠️  RESULT TYPE: Empty response stream")
    logger.info("━" * 60)

    return {
        "final_output": final_payload,
    }

