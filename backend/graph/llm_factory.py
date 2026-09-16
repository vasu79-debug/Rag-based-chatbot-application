"""
LLM Factory for Demo 4.
Instantiates chat model instances based on settings (Groq, OpenAI, OpenRouter, Ollama).
"""

from langchain_core.language_models.chat_models import BaseChatModel
from config import settings


def get_chat_model(temperature: float = None, max_tokens: int = 1500) -> BaseChatModel:
    """Returns an initialized LangChain ChatModel instance."""
    temp = temperature if temperature is not None else settings.AI_TEMPERATURE
    provider = (settings.AI_PROVIDER or "groq").lower()

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.AI_DEFAULT_MODEL,
            groq_api_key=settings.GROQ_API_KEY,
            temperature=temp,
            max_tokens=max_tokens,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.AI_DEFAULT_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=temp,
            max_tokens=max_tokens,
        )
    elif provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.AI_DEFAULT_MODEL,
            openai_api_key=settings.OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=temp,
            max_tokens=max_tokens,
        )
    else:
        # Default fallback to Groq
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=settings.GROQ_API_KEY,
            temperature=temp,
            max_tokens=max_tokens,
        )
