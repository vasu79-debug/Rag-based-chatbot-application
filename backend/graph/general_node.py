import logging
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from config import settings
from graph.state import AgentState
from graph.llm_factory import get_chat_model

logger = logging.getLogger("demo4.graph.general")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


GENERAL_PROMPT = """You are the General Intelligence and Reasoning specialist for an enterprise AI assistant.
Your task is to provide helpful, comprehensive, and intelligent responses using general knowledge, reasoning, industry best practices, external comparisons, or creative drafting (e.g. drafting emails, writing code, explaining laws).

GUIDELINES:
- Provide high-quality, practical, and clear insights.
- If asked to draft an email, document, or code, provide a ready-to-use template.
- If asked for comparison or general background (e.g., industry standards, statutory regulations), explain clearly.
- Maintain a professional, polite, and engaging tone.
- Take into account the conversation history when answering follow-up questions.
"""


def general_node(state: AgentState) -> Dict[str, Any]:
    """Generates the general intelligence and reasoning stream with multi-turn memory and step logs."""
    question = state["question"]
    route = state.get("route", "GENERAL")
    history = state.get("history", [])

    logger.info("━" * 60)
    logger.info(f"🌐 [Step 3/4: GENERAL LLM STREAM] Generating World Knowledge / Reasoning / Actions")
    logger.info(f"   Mode: {route} | Memory Context: {len(history)} turns")

    llm = get_chat_model(temperature=0.3, max_tokens=1000)

    # Construct conversation history window
    messages: List[BaseMessage] = [SystemMessage(content=GENERAL_PROMPT)]

    max_turns = getattr(settings, "MAX_HISTORY_TURNS", 8)
    recent_history = history[-max_turns:] if len(history) > max_turns else history

    for turn in recent_history:
        role = turn.get("role", "").lower()
        content = turn.get("content", "").strip()
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    # Add current user turn
    messages.append(HumanMessage(content=f"User Request: {question}\nContext Mode: {route}"))

    try:
        response = llm.invoke(messages)
        general_answer = response.content.strip()
        logger.info(f"   ✅ [General LLM Stream] Reasoning Output Generated ({len(general_answer)} chars)")
    except Exception as e:
        logger.error(f"   ❌ [General LLM Stream] Error: {str(e)}")
        general_answer = f"Error generating general reasoning: {str(e)}"

    logger.info("━" * 60)

    return {
        "general_answer": general_answer,
    }


