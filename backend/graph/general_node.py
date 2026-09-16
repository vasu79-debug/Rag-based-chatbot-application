"""
General Knowledge & Reasoning Node for Demo 4 LangGraph.
Provides world intelligence, industry best practices, drafting, coding, and creative reasoning.
"""

from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from graph.state import AgentState
from graph.llm_factory import get_chat_model


GENERAL_PROMPT = """You are the General Intelligence and Reasoning specialist for an enterprise AI assistant.
Your task is to provide helpful, comprehensive, and intelligent responses using general knowledge, reasoning, industry best practices, external comparisons, or creative drafting (e.g. drafting emails, writing code, explaining laws).

GUIDELINES:
- Provide high-quality, practical, and clear insights.
- If asked to draft an email, document, or code, provide a ready-to-use template.
- If asked for comparison or general background (e.g., industry standards, statutory regulations), explain clearly.
- Maintain a professional, polite, and engaging tone.
"""


def general_node(state: AgentState) -> Dict[str, Any]:
    """Generates the general intelligence and reasoning stream."""
    question = state["question"]
    route = state.get("route", "GENERAL")

    llm = get_chat_model(temperature=0.3, max_tokens=1000)

    try:
        response = llm.invoke([
            SystemMessage(content=GENERAL_PROMPT),
            HumanMessage(content=f"User Request: {question}\nContext Mode: {route}"),
        ])
        general_answer = response.content.strip()
    except Exception as e:
        general_answer = f"Error generating general reasoning: {str(e)}"

    return {
        "general_answer": general_answer,
    }
