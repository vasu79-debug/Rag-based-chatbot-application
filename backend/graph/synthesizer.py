import logging
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from graph.state import AgentState
from graph.llm_factory import get_chat_model

logger = logging.getLogger("demo4.graph.synthesizer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


SYNTHESIZER_PROMPT = """You are an enterprise AI assistant for Krify (Krify Software Technologies).
Your task is to merge internal organizational knowledge (from company documents) and general intelligence/reasoning into a SINGLE, unified, professional, and clear answer to the user's question.

RULES:
1. Provide ONE seamless, direct, and well-structured answer.
2. DO NOT use separated section headers like "Organisational Knowledge" vs "General Knowledge" or say "From documents... and from AI...".
3. For questions about Krify, its services, team, or policies, strictly prioritize the factual company document excerpts.
4. Seamlessly answer any follow-up, conversational, or general parts of the user's question.
5. Format cleanly using standard Markdown with concise bullet points or headings where helpful.
"""


def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """Combines RAG and General answers into a single cohesive unified response and logs the result."""
    question = state.get("question", "")
    route = state.get("route", "HYBRID")
    rag_answer = state.get("rag_answer")
    rag_available = state.get("rag_available", False)
    citations = state.get("citations", [])
    general_answer = state.get("general_answer")

    unified_content = ""

    if route == "OUT_OF_SCOPE":
        unified_content = (
            "I am an enterprise AI assistant dedicated to helping with Krify company knowledge, "
            "uploaded documents, services, and workplace operations.\n\n"
            "I cannot answer unrelated outside topics such as celebrity trivia, sports, entertainment, or general world trivia. "
            "Please ask a question related to company documents, services, or workplace tasks!"
        )

    elif route == "HYBRID":
        # If both streams produced output, synthesize them into a single cohesive response
        if rag_available and rag_answer and general_answer:
            logger.info("⚡ [Synthesizer] Synthesizing RAG & General streams into a single unified answer...")
            llm = get_chat_model(temperature=0.2, max_tokens=1500)
            try:
                response = llm.invoke([
                    SystemMessage(content=SYNTHESIZER_PROMPT),
                    HumanMessage(content=f"User Question: {question}\n\nCompany Document Excerpts & Findings:\n{rag_answer}\n\nGeneral Intelligence & Context:\n{general_answer}"),
                ])
                res_txt = response.content.strip() if response and response.content else ""
                unified_content = res_txt or rag_answer or general_answer
            except Exception as e:
                logger.warning(f"⚠️  [Synthesizer] LLM synthesis failed, merging directly: {e}")
                unified_content = f"{rag_answer}\n\n{general_answer}"
        elif rag_answer:
            unified_content = rag_answer
        elif general_answer:
            unified_content = general_answer
        else:
            unified_content = "No response could be generated."

    elif route == "RAG":
        if rag_available and rag_answer:
            unified_content = rag_answer
        elif general_answer:
            unified_content = general_answer
        else:
            unified_content = rag_answer or "No matching internal documents were found."

    else:  # Fallback
        unified_content = general_answer or rag_answer or "No response could be generated."


    final_payload = {
        "route": route,
        "content": unified_content,
        "answer": unified_content,
        "citations_count": len(citations),
        "citations": citations,
        # Backward compatibility for any consumers expecting section format:
        "org_section": {
            "label": "Krify Knowledge Base",
            "content": unified_content,
            "citations": citations,
            "has_citations": len(citations) > 0,
        } if len(citations) > 0 else None,
        "general_section": None,
    }

    # Log clearly which stream was retrieved and delivered
    logger.info("━" * 60)
    logger.info("📦 [Step 4/4: SYNTHESIZER] Unified Response Summary:")
    logger.info(f"   Route: {route} | Output Size: {len(unified_content)} chars | Citations: {len(citations)}")
    logger.info(f"   [Final Output]: {unified_content}")
    logger.info("━" * 60)

    return {
        "final_output": final_payload,
    }

