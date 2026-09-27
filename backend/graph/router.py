import logging
from typing import Dict, Any
import json
import re
from langchain_core.messages import SystemMessage, HumanMessage
from graph.state import AgentState
from graph.llm_factory import get_chat_model

logger = logging.getLogger("demo4.graph.router")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


ROUTER_PROMPT = """You are an intelligent query router for an enterprise AI knowledge assistant.
Your job is to analyze the user's input and determine which knowledge sources are required based on the organization's indexed knowledge base (uploaded documents and website content).

Available Categories:
1. "RAG": Factual questions asking about the organization, its services, products, documents, policies, technical guides, specifications, or any information contained in the uploaded knowledge base.
2. "HYBRID": Questions requiring reasoning, analyzing, drafting an email/document, summarizing, comparing, explaining concepts from the knowledge base simply, conversational greetings, OR asking if the company can build custom/new technologies not explicitly listed.
3. "OUT_OF_SCOPE": Queries that are completely unrelated to the organization, business operations, or uploaded documents (e.g. celebrity gossip, sports scores, movie plots, video games, cooking recipes, general trivia).

Examples:
- "Hello, how can you help me?" -> HYBRID
- "What information do you have on [topic from documents]?" -> RAG
- "Which services do you provide to build an Agentic application?" -> HYBRID
- "Summarize section 3 of the uploaded guide" -> HYBRID
- "Draft an email to a client explaining our offerings based on our materials" -> HYBRID
- "Compare [concept A from docs] with [concept B] and give recommendations" -> HYBRID
- "Who won the 2022 FIFA World Cup?" -> OUT_OF_SCOPE
- "Give me a recipe for pizza" -> OUT_OF_SCOPE

Output ONLY a JSON object:
{
  "route": "RAG" | "HYBRID" | "OUT_OF_SCOPE",
  "rag_subquery": "the specific search keywords or question to look up in the knowledge base, or empty string",
  "reasoning": "one short sentence"
}
"""


def router_node(state: AgentState) -> Dict[str, Any]:
    """Analyzes user input and routes to appropriate knowledge branch with structured logging."""
    question = state["question"]
    history = state.get("history", [])
    llm = get_chat_model(temperature=0.0, max_tokens=256)

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
            if route not in ["RAG", "HYBRID", "OUT_OF_SCOPE"]:
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

    if route == "OUT_OF_SCOPE":
        route_icon = "🚫 OUT OF SCOPE (Declining unrelated outside topic)"
    elif route == "RAG":
        route_icon = "🏢 RAG (Internal Docs Only)"
    else:
        route_icon = "⚡ HYBRID (Company Context + LLM Reasoning)"

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


