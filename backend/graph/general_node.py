import logging
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from config import settings
from graph.state import AgentState
from graph.llm_factory import get_chat_model

logger = logging.getLogger("demo4.graph.general")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def build_general_prompt() -> str:
    """Builds the general reasoning prompt dynamically from config settings."""
    assistant_name = getattr(settings, "ASSISTANT_NAME", "Enterprise AI Assistant")
    company_name = getattr(settings, "COMPANY_NAME", "our organization")
    role_desc = getattr(settings, "ASSISTANT_ROLE_DESCRIPTION", "an enterprise AI assistant")
    tone = getattr(settings, "ASSISTANT_TONE", "professional, clear, executive, and helpful")
    custom_rules = getattr(settings, "CUSTOM_SYSTEM_INSTRUCTIONS", "")

    return f"""You are {assistant_name}, {role_desc} for {company_name}.
Your task is to provide intelligent, structured, and helpful responses by applying reasoning, professional drafting, workflow explanations, business comparisons, and summarization to assist with company-related topics, documents, and workplace tasks.

GUIDELINES & TONE:
- Tone: {tone}.
- Provide high-quality, practical, and clear insights for company operations and workplace tasks.
- If asked to draft an email, memo, report, or business template, provide a polished, ready-to-use draft.
- If asked for comparisons, best practices, or explaining complex clauses simply, explain clearly and objectively.
- If a query is completely unrelated to company operations, documents, or workplace tasks, politely steer the user back to company-related topics.
- {custom_rules}
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

    llm = get_chat_model(temperature=0.3, max_tokens=1500)

    # Construct conversation history window dynamically from settings
    prompt_text = build_general_prompt()
    messages: List[BaseMessage] = [SystemMessage(content=prompt_text)]


    max_turns = getattr(settings, "MAX_HISTORY_TURNS", 10)
    recent_history = history[-max_turns:] if len(history) > max_turns else history

    for i, turn in enumerate(recent_history):
        role = turn.get("role", "unknown")
        # Truncate content for clean logging if it's too long
        content = turn.get("content", "").replace("\n", " ")
        log_content = content[:100] + "..." if len(content) > 100 else content
        logger.info(f"   [History Turn {i+1}] {role}: {log_content}")

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
        general_answer = response.content.strip() if response and response.content else ""
        logger.info(f"   ✅ [General LLM Stream] Reasoning Output Generated ({len(general_answer)} chars)")
        logger.info(f"   [General Output]: {general_answer}")
    except Exception as e:
        logger.error(f"   ❌ [General LLM Stream] Error: {str(e)}")
        general_answer = f"Error generating general reasoning: {str(e)}"

    logger.info("━" * 60)

    return {
        "general_answer": general_answer,
    }


