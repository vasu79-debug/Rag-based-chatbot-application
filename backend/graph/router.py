import logging
from typing import Dict, Any
import json
import re
from langchain_core.messages import SystemMessage, HumanMessage
from graph.state import AgentState
from graph.llm_factory import get_chat_model

logger = logging.getLogger("demo4.graph.router")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


ROUTER_PROMPT = """You are an intelligent query router for an enterprise hybrid AI assistant.
Your job is to analyze the user's input and determine which knowledge sources are required.

Available Categories:
1. "GENERAL": The query is a greeting, pleasantry, general world knowledge, math, programming, or creative task that DOES NOT mention or require private internal company documents.
2. "RAG": The query is purely asking about internal company policies, rules, benefits, expenses, guidelines, or uploaded corporate documents.
3. "HYBRID": The query requires BOTH internal company knowledge AND external general reasoning, industry comparison, drafting an email/document, or creative synthesis.

Examples:
- "Hello, how are you?" -> GENERAL
- "What is our company expense reimbursement limit?" -> RAG
- "Compare our annual leave policy with standard UK statutory law and draft an email to HR" -> HYBRID
- "Write a python script to calculate fibonacci" -> GENERAL
- "Based on our security policy, explain best practices for setting up 2FA and give a security checklist" -> HYBRID

Output ONLY a JSON object:
{
  "route": "GENERAL" | "RAG" | "HYBRID",
  "rag_subquery": "the specific question to search in company documents, or empty string",
  "reasoning": "one short sentence"
}
"""


def router_node(state: AgentState) -> Dict[str, Any]:
    """Analyzes user input and routes to appropriate knowledge branch with structured logging."""
    question = state["question"]
    history = state.get("history", [])
    llm = get_chat_model(temperature=0.0, max_tokens=200)

    logger.info("━" * 60)
    logger.info(f"🚦 [Step 1/4: ROUTER] Analyzing Question: \"{question}\"")
    logger.info(f"    History context: {len(history)} turns available")

    # Format brief recent context (last 2 turns) if available
    recent_context = ""
    if history:
        last_turns = history[-2:]
        formatted = "\n".join([f"{t.get('role', 'user')}: {t.get('content', '')[:120]}" for t in last_turns if t.get('content')])
        if formatted:
            recent_context = f"\nRecent Context:\n{formatted}\n"

    reasoning = ""
    try:
        response = llm.invoke([
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=f"{recent_context}Current User Query: {question}"),
        ])
        
        # Parse JSON from response
        content = response.content.strip()
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            route = data.get("route", "HYBRID").upper()
            if route not in ["GENERAL", "RAG", "HYBRID"]:
                route = "HYBRID"
            rag_subquery = data.get("rag_subquery", question) or question
            reasoning = data.get("reasoning", "")
        else:
            route = "HYBRID"
            rag_subquery = question
    except Exception as e:
        logger.warning(f"⚠️  [Router] LLM classification fallback: {e}")
        route = "HYBRID"
        rag_subquery = question

    route_icon = "🌐 GENERAL (Pure LLM)" if route == "GENERAL" else ("🏢 RAG (Internal Docs Only)" if route == "RAG" else "⚡ HYBRID (Dual RAG + LLM)")
    logger.info(f"📍 [Router Decision] ➔ {route_icon}")
    if reasoning:
        logger.info(f"   Reasoning: \"{reasoning}\"")
    if route in ["RAG", "HYBRID"]:
        logger.info(f"   RAG Sub-query: \"{rag_subquery}\"")
    logger.info("━" * 60)

    return {
        "route": route,
        "sub_questions": {"rag": rag_subquery, "general": question},
    }

