"""
Intent Router Node for Demo 4 LangGraph.
Classifies query into GENERAL, RAG, or HYBRID and extracts sub-queries.
"""

from typing import Dict, Any
import json
import re
from langchain_core.messages import SystemMessage, HumanMessage
from graph.state import AgentState
from graph.llm_factory import get_chat_model


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
    """Analyzes user input and routes to appropriate knowledge branch."""
    question = state["question"]
    llm = get_chat_model(temperature=0.0, max_tokens=200)

    try:
        response = llm.invoke([
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=f"User Query: {question}"),
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
        else:
            route = "HYBRID"
            rag_subquery = question
    except Exception:
        # Fallback to HYBRID if router fails
        route = "HYBRID"
        rag_subquery = question

    return {
        "route": route,
        "sub_questions": {"rag": rag_subquery, "general": question},
    }
