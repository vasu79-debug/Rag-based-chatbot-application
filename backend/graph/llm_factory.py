"""
LLM Factory for Demo 4.
Instantiates chat model instances based on settings (Groq, OpenAI, OpenRouter, Ollama).
"""

from langchain_core.language_models.chat_models import BaseChatModel
from config import settings


def get_chat_model(temperature: float = None, max_tokens: int = 1500) -> BaseChatModel:
    """Returns an initialized LangChain ChatModel instance with automatic fallbacks."""
    temp = temperature if temperature is not None else settings.AI_TEMPERATURE
    provider = (settings.AI_PROVIDER or "groq").lower()

    if provider == "groq":
        from langchain_groq import ChatGroq

        primary_model = settings.AI_DEFAULT_MODEL or "openai/gpt-oss-20b"
        main_llm = ChatGroq(
            model=primary_model,
            groq_api_key=settings.GROQ_API_KEY,
            temperature=temp,
            max_tokens=max_tokens,
        )

        # Fallbacks for gpt-oss models
        fallback_models = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
        fallbacks = [
            ChatGroq(
                model=m,
                groq_api_key=settings.GROQ_API_KEY,
                temperature=temp,
                max_tokens=max_tokens,
            )
            for m in fallback_models if m != primary_model
        ]

        return main_llm.with_fallbacks(fallbacks) if fallbacks else main_llm

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        primary_model = settings.AI_DEFAULT_MODEL or "gpt-4o-mini"
        main_llm = ChatOpenAI(
            model=primary_model,
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=temp,
            max_tokens=max_tokens,
        )
        return main_llm.with_fallbacks([
            ChatOpenAI(
                model="gpt-3.5-turbo",
                openai_api_key=settings.OPENAI_API_KEY,
                temperature=temp,
                max_tokens=max_tokens,
            )
        ])

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
            model="openai/gpt-oss-20b",
            groq_api_key=settings.GROQ_API_KEY,
            temperature=temp,
            max_tokens=max_tokens,
        )

